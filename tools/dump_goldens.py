"""Dump the Python bifonia's output as the C++ port's parity goldens.

Every public function is called on every corpus input, and each result is
written as one compact JSON line. The C++ parity test recomputes each line and
compares the bytes.

Files written to --out-dir:
  sentences.jsonl.gz  one line per sentence: tokens, proper flags, the
                      diacritized sentence, and for every homograph token the
                      POS and sense guesses, the four POS scores, every POS
                      and tagger override, and both sense models' scores; last,
                      the sentence diacritized for each lect that has
                      overrides, or null where it equals the default
  forced.jsonl.gz     the same sentences with every model word routed to the
                      model, so the model branch of guess_sense decides
  calls.jsonl.gz      direct calls: tables, vocabularies, text helpers,
                      infopedia, numeric probes, index probes, Unicode probes,
                      lect tags

Sentence corpora:
  --test-sentences  bifonia tests/test_sentences.jsonl (each sentence as given,
                    lowercased, in capitals and with its case swapped)
  --hf-test         bifonia hf/test.jsonl (sentence and diacritized_sentence)
  generated         every homograph in sentence frames, every homograph and
                    diacritized form in mixed case, every .voc term in
                    context, the "molho" clause frames, and a seeded fuzz set
  fixtures          tests/fixtures/molho_probes.tsv (the reading of "molho")

Usage:
  python -I tools/dump_goldens.py --out-dir tests/goldens \\
      --test-sentences <bifonia>/tests/test_sentences.jsonl \\
      --hf-test <bifonia>/hf/test.jsonl
"""
import argparse
import copy
import gzip
import importlib.metadata
import io
import json
import pathlib
import random
import sys
import unicodedata
import zlib

import bifonia
from bifonia import cues, data, features, infopedia, model, scoring, text, vocab

POSES = ("ADP", "NOUN", "VERB", "ADJ")
POSTAGS = ("ADP", "NOUN", "VERB", "ADJ", "PROPN", "AUX", "X")
VOCS = features.load_structural_vocs("pt-pt")
NB = model.SenseModel.load(str(model.NB_PATH))
PERCEPTRON = model.SenseModel.load(str(model.PERCEPTRON_PATH))
LOCALE = pathlib.Path(vocab.__file__).parent / "locale" / "pt-pt"
VOC_NAMES = sorted(p.stem for p in LOCALE.glob("*.voc"))


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # the exception class is part of the contract
        return {"err": type(exc).__name__}


def feature_list(words, idx):
    return [[k, repr(v)] for k, v in features.extract_features(words, idx, VOCS).items()]


def model_record(sense_model, base, words, idx):
    if not sense_model.has(base):
        return None
    scores = sense_model.scores(base, words, idx)
    return [sense_model.route(base), repr(sense_model.margin_tau(base)),
            [[s, repr(v)] for s, v in scores.items()],
            sense_model.predict(base, words, idx), repr(sense_model.margin(base, words, idx))]


def candidates(words):
    for i, tok in enumerate(words):
        token = bifonia._strip_edge(tok)
        if token in data.AMBIGUOUS_WORDS or token in bifonia._DIACRITIZED_TO_BASE:
            yield i, token


# The lects that have rows in data/lect_overrides.csv.
LECTS = ("pt-BR", "pt-UY")


def sentence_record(sentence, detail):
    words = bifonia.tokenize(sentence)
    flags = bifonia.proper_flags(sentence)
    out = [sentence, words, "".join("1" if f else "0" for f in flags),
           "".join("1" if bifonia.is_ambiguous(w) else "0" for w in words),
           call(bifonia.add_extra_diacritics, sentence)]
    cands = []
    for i, token in candidates(words):
        base = bifonia._DIACRITIZED_TO_BASE.get(token, token)
        norm = list(words)
        norm[i] = base
        feats = feature_list(norm, i)
        cands.append([
            i,
            call(bifonia.guess_pos, words, i), call(bifonia.guess_pos, words, i, proper=True),
            call(bifonia.guess_sense, words, i), call(bifonia.guess_sense, words, i, proper=True),
            call(bifonia.disambiguate, words, i),
            [call(f, norm, i) for f in (scoring.score_adp, scoring.score_noun,
                                        scoring.score_verb, scoring.score_adj)],
            [call(scoring.resolve_sense, base, norm, i, p) for p in POSES],
            [call(bifonia.guess_sense, words, i, postag=t) for t in POSTAGS],
            [call(bifonia.disambiguate, words, i, pos=p) for p in POSES],
            model_record(NB, base, norm, i),
            model_record(PERCEPTRON, base, norm, i),
            feats if detail == 2 else zlib.crc32(dumps(feats).encode("utf-8")),
        ])
    out.append(cands if detail else zlib.crc32(dumps(cands).encode("utf-8")))
    default = out[4]
    out.append([None if (r := call(bifonia.add_extra_diacritics, sentence, lang)) == default else r
                for lang in LECTS])
    return out


def forced_model():
    forced = copy.deepcopy(PERCEPTRON)
    for k, entry in enumerate(forced.words.values()):
        entry["route"] = "model"
        entry["margin_tau"] = (k % 3) * 0.75
    return forced


def forced_record(sentence):
    words = bifonia.tokenize(sentence)
    return [sentence, call(bifonia.add_extra_diacritics, sentence),
            [[i, call(bifonia.guess_sense, words, i), call(bifonia.disambiguate, words, i)]
             for i, _ in candidates(words)]]


# ── sentence corpora ──────────────────────────────────────────────────────────

FRAMES = (
    "{w}", "{w}.", "o {w}", "a {w}", "um {w}", "uma {w}", "eu {w}", "ele {w} o carro",
    "não {w} nada", "o {w} do vizinho", "a {w} da casa", "de {w}", "em {w} de",
    "com a {w} dele", "o {w} todo", "ontem, {w} a porta", "que {w} tão bom",
    "{w}-me", "{w}-o", "{w} -lhe", "muito {w}", "é {w}", "foi {w} pelo juiz",
    "para {w}", "quando {w} a bola", "o meu {w} é grande", "Ontem o {w} Silva chegou.",
    "{w} rapidamente as mãos", "{w} para ti", "{w} de ti", "talvez {w} algum",
    "sempre {w} uma parte", "depois de nadar, {w} o cabelo", "o carro {w} no semáforo",
    "{w} 13", "{w} nº 5", "O {w} Corá fica longe", "a {w} e a {w}", "{w}, {w}; {w}!",
    "mais {w} que", "estava {w} a secar", "aprendeu a {w} a fruta",
)

GENERAL_TARGETS = ("para", "pelo", "sobre", "sede", "molho", "tola", "forma", "corte",
                   "colher", "gozo", "posto", "torno", "troco", "seco", "bola", "lobo",
                   "polo", "cor", "choco", "solto", "jogo", "gosto", "acordo", "força")

VOC_TARGETS = {
    "bola_ball_cues": ("bola",), "bola_loaf_cues": ("bola",), "lobo_lobe_cues": ("lobo",),
    "polo_fledgling_cues": ("polo",), "cor_memory_cues": ("cor",),
    "molho_noun_prev": ("molho",), "adverbial_de": ("molho",), "object_pronouns": ("molho",),
    "object_determiners": ("molho",), "a_locutions": ("molho",), "post_verb_adverbs": ("molho",),
    "frequency_nouns": ("molho",), "time_nouns": ("molho",), "degree_nouns": ("molho",),
    "evaluative_nouns": ("molho",), "evaluative_adjectives": ("molho",),
    "duration_modifiers": ("molho",), "por_contractions": ("molho",),
    "mould_cues": ("forma",), "bakeware": ("forma",), "bake_purpose": ("forma",),
    "forma_shape_prev": ("forma",), "forma_shape_next": ("forma",),
    "sede_seat_cues": ("sede",), "sede_thirst_cues": ("sede",),
    "bundle_things": ("molho",), "bundle_ambiguous": ("molho",), "bundle_verbs": ("molho",),
    "soak_verbs": ("molho",), "cut_context": ("corte",), "court_terms": ("corte",),
    "fem_det": ("corte", "tola"), "masc_det": ("corte",), "head_verbs": ("tola",),
    "head_state": ("tola",), "wood_context": ("tola",), "tola_poss_prev": ("tola",),
    "tola_det_prev": ("tola",), "person_possessive": ("tola",), "material_prep": ("tola",),
    "head_locative": ("tola",), "pelo_fixed": ("pelo",), "pelo_route": ("pelo",),
    "possession_verbs": ("pelo", "gosto"), "stoppable_things": ("para",),
    "para_verb_next": ("para",), "para_prev_subj": ("para",), "deferral_adv": ("para",),
    "pure_negation": ("para",), "sobre_governors": ("sobre",), "sobre_intensifiers": ("sobre",),
    "sobre_adp_quant": ("sobre",), "optative_adv": ("sobre",), "colher_control_verbs": ("colher",),
    "prep_governing": ("gozo",), "passive_aux": ("posto", "acordo"), "around_prep": ("torno",),
}

VOC_FRAMES = (
    "{t} {w}", "{w} {t}", "o {w} {t}", "{w} de {t}", "{t} a {w} o carro", "{t}, {w}",
    "{w} a {t}", "{w} o {t}", "{t} {x} {w}", "{w} {x} {t}", "{t} o {w} dele",
    "a {w} para {t}", "{t} que {w}", "{w} -{t}", "na {w} {t}", "pela {w} {t}",
)


def variants(word):
    out = [word, word.capitalize(), word.upper()]
    out += [v for (base, _), v in bifonia._DIACRITIZED.items() if base == word]
    out += [v.capitalize() for (base, _), v in bifonia._DIACRITIZED.items() if base == word]
    return list(dict.fromkeys(out))


def frame_sentences():
    for word in data.HOMOGRAPHS:
        for v in variants(word):
            for frame in FRAMES:
                yield frame.format(w=v)


# Frames that steer different senses, with the homograph first and inside.
CASE_FRAMES = (
    "{w}", "{w} do partido", "A {w} DO PARTIDO FICA NO PORTO.", "tenho muita {w}",
    "O {w} foi assinado", "eu {w} cedo", "{w}, {w}!", "não {w} AQUI",
)


def case_variants(word):
    """The word in capitals and in three mixed-case patterns."""
    return [word.upper(),
            "".join(c.upper() if k % 2 == 0 else c for k, c in enumerate(word)),
            "".join(c.upper() if k % 2 else c for k, c in enumerate(word)),
            word[:-1] + word[-1:].upper()]


def case_sentences():
    """add_extra_diacritics gives each mark the case of its letter."""
    for word in data.HOMOGRAPHS:
        forms = [word] + [v for (base, _), v in bifonia._DIACRITIZED.items() if base == word]
        for v in dict.fromkeys(c for f in forms for c in case_variants(f)):
            for frame in CASE_FRAMES:
                yield frame.format(w=v)


def voc_sentences():
    others = [w for w in data.HOMOGRAPHS if w not in GENERAL_TARGETS]
    k = 0
    for name in VOC_NAMES:
        for term in sorted(vocab.voc(name)):
            targets = VOC_TARGETS.get(name)
            if targets is None:
                targets = (GENERAL_TARGETS[k % len(GENERAL_TARGETS)],
                           GENERAL_TARGETS[(k * 7 + 3) % len(GENERAL_TARGETS)],
                           others[k % len(others)])
            k += 1
            for w in targets:
                for frame in VOC_FRAMES:
                    yield frame.format(t=term, w=w, x="o")
            yield term.capitalize() + " " + targets[0]
            yield text.fold(term) + " " + targets[0] + " " + text.fold(term)


def extra_sentences():
    """Rule branches that the other corpora do not reach."""
    # the locative contractions before sede
    for prep in ("na", "à", "numa", "pela", "duma", "da", "Duma", "DUMA"):
        for tail in ("", " de água", " amor", " de poder", " da empresa", " tenho"):
            yield f"{prep} sede{tail}"
    # a left neighbour whose last three code points are not three bytes
    for left in ("çia", "ãia", "çía", "ßia", "xia", "eia", "ia", "çiã"):
        for w in ("para", "sede", "molho", "bola", "corte", "forma", "pelo"):
            yield f"{left} {w} de água"
    # proper_flags does not lowercase and tokenize does: İ splits a token, the
    # flags shift, and a replacement lands on the next word
    for h in ("seco", "para", "sede", "molho", "pelo", "corte", "acordo", "jogo"):
        for t in ("ǅ", "Ǆ", "ǆ", "ᾈ", "ᾀ", "Σ", "É", "ß", "x", "X", "1", "ǲ", "Ǳ", "ﬁ", "Ⅻ", "ⅻ", "ǈ"):
            yield f"İ{h} {t} x"
            yield f"x İ{h} {t}"


# The lists that decide whether "molho" opens a clause with its object.
MOLHO_VOCS = ("molho_noun_prev", "adverbial_de", "object_pronouns", "object_determiners",
              "a_locutions", "post_verb_adverbs", "frequency_nouns", "time_nouns", "degree_nouns",
              "evaluative_nouns", "evaluative_adjectives", "duration_modifiers", "por_contractions",
              "adv_bridge", "neg_adv", "deferral_adv", "intensifiers", "articles", "quantifiers",
              "numeric", "determiners")

MOLHO_FRAMES = (
    "Molho {t}", "Molho {t} as plantas", "Molho {t}.", "Molho {t}, as plantas", "molho {t} o pão",
    "Ontem, molho {t} as plantas", "eu molho {t}", "{t} molho", "{t} molho as plantas",
    "o {t} molho", "Molho um {t} de pão", "Molho uma {t}!", "Molho uma bela {t}",
    "Molho todo o {t}", "Molho uma {t} inteira as plantas", "Molho de {t} as plantas",
    "Molho a {t} os vasos", "Molho duas {t} os vasos", "Molho {t}-o", "Molho {t} -as",
    "o molho de {t}", "a qual molho {t}", "{t} qual molho", "pelo qual molho {t}",
    "Molho sempre {t} as plantas", "Molho mais {t}", "Molho {t} todo-o-terreno",
    "Qual molho, {t}?", "Molho {t} {t} as plantas", "falta {t}; molho {t} os pés",
)


def molho_sentences():
    """Each term of the lists that score "molho", in the clause frames that read them."""
    for name in MOLHO_VOCS:
        for term in sorted(vocab.voc(name)):
            for frame in MOLHO_FRAMES:
                yield frame.format(t=term)


MOLHO_PROBES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "molho_probes.tsv"


def molho_probe_sentences():
    for line in MOLHO_PROBES.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            yield line.split("\t", 1)[1]


FUZZ_PUNCT = (".", ",", ";", ":", "!", "?", "-", " -", "...", "!?", "…", "«", "»", "'", '"',
              "(", ")", "—", "–", "_")
FUZZ_EXTRA = ("İstanbul", "ǅemal", "Σίσυφος", "ΣΑΣ", "ß", "ﬁm", "１２", "١٢", "²", "Ⅻ", "1_000",
              "1e5", "-3", "nan", "inf", "Infinity", "1__0", "_1", "x²", "nº", "n.º", "NO", "Nr",
              "água", "páscoa", "Pascoa", "chouriço", "chourico", " ", "\t", "\n",
              "ȩ́", "ǰ", "ﬀ", "Å", "Ω", "K", "ﾊ", "가", "à", "Às", "ÀS", "aﾞ")


def fuzz_sentences(count, seed):
    rng = random.Random(seed)
    homographs = list(data.HOMOGRAPHS)
    diacritized = list(bifonia._DIACRITIZED.values())
    terms = sorted({t for name in VOC_NAMES for t in vocab.voc(name)})
    for _ in range(count):
        parts = []
        for _ in range(rng.randint(1, 10)):
            r = rng.random()
            if r < 0.3:
                tok = rng.choice(homographs)
            elif r < 0.37:
                tok = rng.choice(diacritized)
            elif r < 0.75:
                tok = rng.choice(terms)
            elif r < 0.85:
                tok = rng.choice(FUZZ_EXTRA)
            else:
                tok = "".join(chr(rng.choice((rng.randint(0x20, 0x17F), rng.randint(0x370, 0x3FF),
                                              rng.randint(0x300, 0x36F))))
                              for _ in range(rng.randint(1, 5)))
            c = rng.random()
            if c < 0.15:
                tok = tok.capitalize()
            elif c < 0.2:
                tok = tok.upper()
            if rng.random() < 0.25:
                tok += rng.choice(FUZZ_PUNCT)
            if rng.random() < 0.06:
                tok = rng.choice(FUZZ_PUNCT) + tok
            parts.append(tok)
        sep = rng.choice((" ", " ", " ", "  ", "-", " - ", "\t", ", "))
        yield sep.join(parts)


# ── direct calls ──────────────────────────────────────────────────────────────

def table_calls():
    yield ["HOMOGRAPHS", list(data.HOMOGRAPHS.items())]
    yield ["SENSE_POS", list(data.SENSE_POS.items())]
    yield ["SENSE_POSES", [[w, [[s, list(p)] for s, p in d.items()]] for w, d in data.SENSE_POSES.items()]]
    yield ["POS_SENSES", list(data.POS_SENSES.items())]
    for name in ("AMBIGUOUS_WORDS", "POS_AMBIGUOUS", "STOPPABLE_THINGS", "DET", "QUANT", "PRON",
                 "AUX_VERBS", "NUMERIC", "SOBRE_GOV", "NEVER_AFTER_PREP", "PREP", "BEFORE_PREP",
                 "AFTER_PREP"):
        yield [name, sorted(getattr(data, name))]
    for name in ("VERBS_IPA", "NOUNS_IPA", "ADP_IPA", "ADJ_IPA", "DEFAULT_SENSE", "DEFAULT_POS",
                 "BASE_SCORE"):
        yield [name, list(getattr(data, name).items())]
    yield ["SENSE_CUES", [[w, r.default, [[c.sense, c.voc, c.requires_prev] for c in r.cues]]
                          for w, r in cues.SENSE_CUES.items()]]
    yield ["FEATURE_CUES", [[k, list(v)] for k, v in cues.FEATURE_CUES.items()]]
    yield ["DIACRITIZED", [[b, s, f] for (b, s), f in bifonia._DIACRITIZED.items()]]
    yield ["STRUCTURAL_VOCS", list(features.STRUCTURAL_VOCS)]
    yield ["DEFAULT_LANG", vocab.DEFAULT_LANG]
    yield ["NB_PATH", model.NB_PATH.name]
    yield ["PERCEPTRON_PATH", model.PERCEPTRON_PATH.name]
    yield ["VERSION", bifonia.version.VERSION_STR]
    for lang in ("pt-pt", "PT-PT", "pt", "xx"):
        yield ["load_structural_vocs", lang,
               [[k, sorted(v)] for k, v in features.load_structural_vocs(lang).items()]]
    for name in VOC_NAMES + ["missing_voc"]:
        for lang in ("pt-PT", "pt_PT", "pt", "PT-BR", "en"):
            result = call(vocab.voc, name, lang)
            yield ["voc", name, lang, result if isinstance(result, dict) else sorted(result)]
        if name != "missing_voc":
            yield ["folded", name, sorted(text.folded(vocab.voc(name)))]
    for path, m in (("nb", NB), ("perceptron", PERCEPTRON)):
        yield ["model", path, m.version, m.model_type, m.lang,
               [[k, repr(v) if isinstance(v, float) else v] for k, v in m.config.items()]]
        for word, entry in m.words.items():
            weights = [[s, len(entry["weights"].get(s, {})),
                        zlib.crc32(dumps(sorted((f, repr(v)) for f, v in entry["weights"].get(s, {}).items())).encode())]
                       for s in entry["senses"]]
            yield ["model_word", path, word, m.has(word), m.route(word), repr(m.margin_tau(word)),
                   entry["senses"], [[s, repr(v)] for s, v in entry["bias"].items()], weights]
        for word in ("", "xyz", "Para", "pára"):
            yield ["model_word_absent", path, word, m.has(word), m.route(word), repr(m.margin_tau(word))]


def infopedia_calls():
    rows = infopedia.load()
    yield ["infopedia.load", [[w, [list(r.items()) for r in rs]] for w, rs in rows.items()]]
    words = list(rows) + [w.upper() for w in rows] + [w.capitalize() for w in rows] + \
        ["", "xyz", "PÁRA", "İ", "ΣΑΣ"]
    for w in words:
        yield ["infopedia.readings_len", w, len(infopedia.readings(w))]
        for pos in (None, "", "VERB", "NOUN", "ADJ", "ADP", "V", "N", "verbo", "nome",
                    "NOUN|VERB", "VERB|NOUN", "|", "NOUN|", "|NOUN"):
            yield ["infopedia.ipa", w, pos, infopedia.ipa(w, pos)]


NUMBERS = ("0", "1", "12", "-1", "+1", "1.5", ".5", "5.", "1e5", "1E5", "1e+5", "1e-5", "1e", "e5",
           "1_000", "1__000", "_1", "1_", "1_.5", "1._5", "1e_5", "1_e5", "1e1_0", "inf", "Inf",
           "-inf", "+inf", "infinity", "INFINITY", "-Infinity", "infin", "nan", "NaN", "-nan", "+nan",
           "nan0", "١٢", "１２", "١.٥", "²", "Ⅻ", "½", "x1", "1x", "0x10", "1e999", "-1e999", "00",
           "0_0", "١_٢", "1 ", " 1", "1 ", "dois", "três", "mil", "três.", "-", "", ".", "-.5",
           "1.5e3", "1.e3", ".e3", "1,5", "um", "uma", "1e-400", "١e٢", "٣٫٤",
           "\x0b1", "1\x0c", "\r1\r", "\t1\n", "\x0b\x0c1.5\r\n", "\x1c1", "1\x1f", "1\x85",
           "\u20281", "\xa01", "1\u3000", "\x0b", "\r\n",
           "--1", "-١٢", "−1", "-²", "-Ⅻ", "1-", "-1-", "-0", "- 1", "-dois", "-três")


def text_calls():
    probes = set()
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        nontrivial = (c.lower() != c or c.upper() != c or c.isupper() or c.isdigit()
                      or c.isspace() or unicodedata.decimal(c, -1) >= 0
                      or unicodedata.normalize("NFD", c) != c or unicodedata.combining(c)
                      or unicodedata.category(c) == "Mn")
        if nontrivial:
            probes.add(c)
    # the boundaries of every \w range
    prev = False
    for cp in range(0x110001):
        cur = cp < 0x110000 and not (0xD800 <= cp <= 0xDFFF) and (chr(cp).isalnum() or chr(cp) == "_")
        if cur != prev:
            for x in (cp - 1, cp):
                if 0 <= x < 0x110000 and not (0xD800 <= x <= 0xDFFF):
                    probes.add(chr(x))
        prev = cur
    for c in sorted(probes):
        s = "a" + c + "b"
        yield ["unicode", c, bifonia.tokenize(s), bifonia.tokenize(c + "para"),
               "".join("1" if f else "0" for f in bifonia.proper_flags("x " + c + "y " + c)),
               text.fold(s), text.strip_punct(s), bifonia.is_ambiguous(c),
               call(bifonia.add_extra_diacritics, c + "Para " + c + "o carro")]
    for s in ("para", "Pára!", "«para»", "(sobre)", "--para--", "pa.ra", "—torno—", "ȩ́",
              "ḍ̇", "q̣̇", "ཱཱི", "\U0001D15F\U0001D165\U0001D16E",
              "가각", "각", "Ǻ", ".,;:!?\"'()[]{}«»–—", "ǅ", "ﬁ", "",
              # non-Mn combining marks out of canonical order: fold keeps them reordered
              "\U0001D16D\U0001D165", "a\U0001D16D\U0001D165b", "\u302E\u1715",
              "\U00016FF0\u302E\u1734", "x\u0307\U0001D16D\u0323\U0001D165"):
        yield ["text", s, text.strip_punct(s), text.fold(s), bifonia.tokenize(s),
               "".join("1" if f else "0" for f in bifonia.proper_flags(s)),
               bifonia.is_ambiguous(s)]
    for n in NUMBERS:
        yield ["score_adp", ["para", n], 0, call(scoring.score_adp, ["para", n], 0)]
    for words, cue_voc in ((["bola", "de", "futebol", "golo"], "bola_ball_cues"),
                           (["a", "páscoa", "bola", "x", "y", "z", "w", "chourico"], "bola_loaf_cues"),
                           (["bola"], "bola_ball_cues")):
        cue_set = text.folded(vocab.voc(cue_voc))
        for idx in range(-len(words) - 1, len(words) + 2):
            for near in (0, 1, 4, 10):
                yield ["cue_score", words, idx, cue_voc, near, text.cue_score(words, idx, cue_set, near)]


def index_calls():
    sentences = ("o autocarro para em frente", "eu gosto de ti", "a sede da empresa",
                 "para", "", "sede de poder, pára!", "a forma de bolo",
                 "molho sempre as plantas", "falta molho, molho um pouco de pão", "qual molho",
                 "o chôco frito", "chôco")
    for s in sentences:
        words = bifonia.tokenize(s)
        for idx in range(-len(words) - 2, len(words) + 2):
            yield ["index", words, idx,
                   call(bifonia.guess_pos, words, idx), call(bifonia.guess_sense, words, idx),
                   call(bifonia.guess_sense, words, idx, pos="NOUN"),
                   call(bifonia.guess_sense, words, idx, pos=""),
                   call(bifonia.disambiguate, words, idx),
                   call(scoring.guess_pos, words, idx),
                   [call(f, words, idx) for f in (scoring.score_adp, scoring.score_noun,
                                                  scoring.score_verb, scoring.score_adj)],
                   call(lambda: feature_list(words, idx))]
    # explicit overrides and errors on fixed sentences
    for s in ("a sede da empresa", "eu jogo às cartas", "o molho de chaves", "a tola dele"):
        words = bifonia.tokenize(s)
        for i in range(len(words)):
            for sense in ("seat", "thirst", "play", "game", "bundle", "sauce", "head", "", "nope"):
                yield ["disambiguate_sense", words, i, sense, call(bifonia.disambiguate, words, i, sense=sense)]
            for pos in ("NOUN", "VERB", "ADP", "ADJ", "", "X"):
                for word in ("sede", "jogo", "molho", "tola", "xyz", ""):
                    yield ["resolve_sense", word, words, i, pos, call(scoring.resolve_sense, word, words, i, pos)]


ODD_STRINGS = (
    "", " ", "-", "--", "_", "-_", "para-", "-para", "--para", "para--o", "PARA", "Pára", "PÁRA",
    "pára!", "para...!?", "para—", "«para»", "ﬁ", "İ", "İpara", "Para para", "nº", "N.º 5",
    "Choro nº 13", "Cerro Corá", "Torno-me", "o -o", "torno-o", "\u2028para", "para\u0301",
    "pa\u0301ra", "a\u00a0para", "1.5", "١٢", "Σ para Σ", "ǅemal para", "e\u0301", "Sede: SEDE",
    "eu Gosto", "O Molho de chaves", "COLHER", "Ação", "...", "a.b,c;d:e!f?g",
    "Choro n.º 13", "Choro N.º 13", "Choro n. º", "Choro n.º", "Choro n.", "Choro no 5", "Choro N.ºx",
)

ODD_WORDS = (
    [], [""], ["para"], ["", "para", ""], ["-me"], ["para,"], ["Para"], ["pára"], ["séde", "de"],
    ["sede", "de", "poder"], ["o", "molho", "de", "salsa"], ["xyz"], ["-", "-"], ["para", "-se"],
    ["bola", "", "golo"], ["cor", "de"], ["de", "cor"], ["a", "tola", "dele"], ["pelo", "o"],
    ["o", "corte", "de", "cabelo"], ["forma", "para", "o", "bolo"],
    ["molho", ""], ["", "molho", "um", ""], [",", "molho"], ["molho", "-as"],
    ["molho", "um", "pouco", "de"], ["x,", "molho", "todo", "o", "dia", "inteiro", ""],
    ["chôco"], ["o", "chôco", "frito"], ["", "molho", "x"], ["", "molho", "de", "tomate"],
)

VOCS_ARGS = ([], [["determiners", ["o", "a"]], ["empty", []], ["x", ["para", "de", ""]]])

CUE_LISTS = ([], ["golo", "Páscoa", "ação", ""], ["de"])


def as_vocs(pairs):
    return {k: frozenset(v) for k, v in pairs}


def result(value):
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        return [[k, result(v)] for k, v in value.items()]
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [result(v) for v in value]
    return value


def probe(fn, *args):
    try:
        return result(fn(*args))
    except Exception as exc:
        return {"err": type(exc).__name__}


def model_of(name):
    return NB if name == "nb" else PERCEPTRON


def probe_calls():
    """Every public function on hand-built arguments, errors included."""
    for s in ODD_STRINGS:
        yield ["probe", "tokenize", [s], probe(bifonia.tokenize, s)]
        yield ["probe", "proper_flags", [s], probe(bifonia.proper_flags, s)]
        yield ["probe", "add_extra_diacritics", [s], probe(bifonia.add_extra_diacritics, s)]
        yield ["probe", "is_ambiguous", [s], probe(bifonia.is_ambiguous, s)]
        yield ["probe", "strip_punct", [s], probe(text.strip_punct, s)]
        yield ["probe", "fold", [s], probe(text.fold, s)]
        yield ["probe", "infopedia.readings", [s], probe(infopedia.readings, s)]
        for pos in (None, "", "VERB"):
            yield ["probe", "infopedia.ipa", [s, pos], probe(infopedia.ipa, s, pos)]
        for lang in ("pt-PT", ""):
            yield ["probe", "voc", [s, lang], probe(vocab.voc, s, lang)]
        yield ["probe", "load_structural_vocs", [s], probe(features.load_structural_vocs, s)]
        for name in ("nb", "perceptron"):
            for fn in ("has", "route", "margin_tau"):
                yield ["probe", "model." + fn, [name, s], probe(getattr(model_of(name), fn), s)]
    for cues_list in CUE_LISTS:
        yield ["probe", "folded", [cues_list], probe(text.folded, frozenset(cues_list))]
    for name in ("missing.json", "heterophonic_homographs.csv", "sense_model_nb.json"):
        yield ["probe", "model.load", [name],
               probe(lambda n: model.SenseModel.load(str(model._DATA / n)).model_type, name)]
    for word in cues.SENSE_CUES:
        rule = cues.SENSE_CUES[word]
        yield ["probe", "cue_repr", [word], [repr(rule)] + [repr(c) for c in rule.cues]]
    for words in ODD_WORDS:
        n = len(words)
        for idx in range(-n - 2, n + 2):
            for cues_list in CUE_LISTS:
                for near in (0, 4, -1):
                    yield ["probe", "cue_score", [words, idx, cues_list, near],
                           probe(text.cue_score, words, idx, frozenset(cues_list), near)]
            for name, fn in (("score_adp", scoring.score_adp), ("score_noun", scoring.score_noun),
                             ("score_verb", scoring.score_verb), ("score_adj", scoring.score_adj)):
                yield ["probe", name, [words, idx], probe(fn, words, idx)]
            for proper in (False, True):
                yield ["probe", "scoring.guess_pos", [words, idx, proper], probe(scoring.guess_pos, words, idx, proper)]
                yield ["probe", "guess_pos", [words, idx, proper], probe(bifonia.guess_pos, words, idx, proper)]
            for pos, proper, postag in ((None, False, None), ("", False, None), ("X", False, None),
                                        ("VERB", True, None), (None, True, None), (None, False, ""),
                                        (None, False, "NOUN"), (None, True, "PROPN"), ("ADJ", False, "AUX")):
                yield ["probe", "guess_sense", [words, idx, pos, proper, postag],
                       probe(bifonia.guess_sense, words, idx, pos, proper, postag)]
            for pos, sense, postag in ((None, None, None), ("X", None, None), (None, "", None),
                                       (None, "thirst", None), ("NOUN", None, "VERB")):
                yield ["probe", "disambiguate", [words, idx, pos, sense, postag],
                       probe(bifonia.disambiguate, words, idx, pos, sense, postag)]
            for word in ("", "sede", "molho", "tola", "bola", "cor", "peso", "para", "xyz"):
                for pos in ("", "NOUN", "VERB", "X"):
                    yield ["probe", "resolve_sense", [word, words, idx, pos],
                           probe(scoring.resolve_sense, word, words, idx, pos)]
            for vocs_arg in VOCS_ARGS:
                yield ["probe", "extract_features", [words, idx, vocs_arg],
                       probe(features.extract_features, words, idx, as_vocs(vocs_arg))]
            for name in ("nb", "perceptron"):
                for word in ("sede", "bola", "xyz"):
                    for vocs_arg in (None,) + VOCS_ARGS[1:]:
                        vocs_value = None if vocs_arg is None else as_vocs(vocs_arg)
                        for fn in ("scores", "predict", "margin"):
                            yield ["probe", "model." + fn, [name, word, words, idx, vocs_arg],
                                   probe(getattr(model_of(name), fn), word, words, idx, vocs_value)]


BYTE_PROBES = ["e282", "e28241", "e282ac", "c0af", "c1bf", "c280", "c2", "df", "dfbf", "e0", "e080",
               "e09f80", "e0a080", "ed9fbf", "eda080", "edbfbf", "ee8080", "f0", "f08f8080",
               "f0908080", "f48f8080", "f4908080", "f5808080", "ff", "fe", "80", "bf", "8080",
               "f09080", "f0908041", "e2", "e2e282ac", "c3a1", "c3", "c3c3a1"]


def byte_calls():
    """Ill-formed UTF-8: the C++ reads raw bytes as Python's decode('utf-8', 'replace')."""
    probes = list(BYTE_PROBES)
    for b in range(0x80, 0x100):
        probes.append(f"{b:02x}")
        for c in (0x41, 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf, 0xc0):
            probes.append(f"{b:02x}{c:02x}")
            probes.append(f"{b:02x}{c:02x}80")
            probes.append(f"{b:02x}80{c:02x}80")
    for h in dict.fromkeys(probes):
        for frame in ("{}", "70617261{}", "{}70617261", "4f20{}6175746f636172726f2070617261"):
            hx = frame.format(h)
            s = bytes.fromhex(hx).decode("utf-8", "replace")
            words = bifonia.tokenize(s)
            yield ["bytes", hx, words, "".join("1" if f else "0" for f in bifonia.proper_flags(s)),
                   call(bifonia.add_extra_diacritics, s), bifonia.is_ambiguous(s), text.fold(s),
                   text.strip_punct(s), [call(bifonia.guess_sense, words, i) for i in range(len(words))]]


# word lists and cue sets as hex: a set member with ill-formed UTF-8 compares as
# the string Python decodes it to
BYTE_SETS = ((["c3", "78", "c3"], ["c3"]), (["efbfbd", "78", "c3a1"], ["c3", "e1"]),
             (["ff", "78", "fe"], ["efbfbd"]), (["e282", "78"], ["e282ac", "e2"]),
             (["4f", "c3c3", "78"], ["c3c3", "efbfbdefbfbd"]), (["f09080", "78", "f0908041"], ["f09080", "f0"]),
             (["c3a1", "78"], ["61cc81"]), (["2ec3", "78", "c32e"], ["c3"]))


def byte_set_calls():
    for words_hex, cues_hex in BYTE_SETS:
        words = [bytes.fromhex(h).decode("utf-8", "replace") for h in words_hex]
        cue_set = frozenset(bytes.fromhex(h).decode("utf-8", "replace") for h in cues_hex)
        for idx in range(len(words)):
            yield ["bytes_sets", words_hex, cues_hex, idx, text.cue_score(words, idx, cue_set, 4),
                   result(features.extract_features(words, idx, {"v": cue_set}))]


CUE_TEXTS = ("a\nb", "\t", "\x7f", "\x85", "\x00", "\x1b", "'", '"', "'\"", "\\", "é", "\u2028", "\u00ad",
             "\u200b", "\U000e0001", "\u0378", "\ue000", "\u3000", "\xa0", " ", "\U0001F600", "\uffff",
             "\U0010ffff", "x\ry", "ǅ", "", "\u0300", "\U00016FF0")


def cue_repr_calls():
    for i, t in enumerate(CUE_TEXTS):
        other = CUE_TEXTS[(i * 5 + 3) % len(CUE_TEXTS)]
        for prev in (None, t, other):
            cue = cues.Cue(t, other, prev)
            yield ["cue_repr_custom", t, other, prev, repr(cue),
                   repr(cues.CueRule(other, (cue,))), repr(cues.CueRule(t, ())),
                   repr(cues.CueRule(t, (cue, cues.Cue(other, t))))]


FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FIXTURE_SENTENCES = ((["o", "para"], 1), (["o", "autocarro", "para", "aqui"], 2),
                     (["a", "sede", "de", "água"], 1), (["o", "molho", "de", "chaves"], 1),
                     (["a", "bola", "de", "futebol"], 1), (["para"], 0), ([], 0))


def no_surrogates(value):
    """A lone surrogate, which the port reads as U+FFFD (a documented divergence)."""
    if isinstance(value, str):
        return "".join("\ufffd" if 0xD800 <= ord(c) < 0xE000 else c for c in value)
    if isinstance(value, list):
        return [no_surrogates(v) for v in value]
    if isinstance(value, dict):
        return {no_surrogates(k): no_surrogates(v) for k, v in value.items()}
    return value


def fixture_calls():
    for record in fixture_records():
        yield no_surrogates(record)


def fixture_records():
    """Model files that test the JSON reader and the model arithmetic at its edges.

    Scores and margins are compared as floats: Python keeps an integer score as
    an int, which prints without ".0" (a documented divergence)."""
    default_model = bifonia._sense_model
    for path in sorted((FIXTURES / "models").glob("*.json")):
        name = path.name
        loaded = call(model.SenseModel.load, str(path))
        if isinstance(loaded, dict):
            yield ["fixture_model", name, loaded]
            continue
        m = loaded
        yield ["fixture_model", name, [m.model_type, m.lang, list(m.words), repr(m.version),
                                       [[k, json.dumps(v)] for k, v in m.config.items()]]]
        as_float = {
            "scores": lambda *a: {k: float(v) for k, v in m.scores(*a).items()},
            "predict": m.predict,
            "margin": lambda *a: float(m.margin(*a)),
        }
        for word in m.words:
            runs = []
            for words, idx in FIXTURE_SENTENCES:
                for fn in ("scores", "predict", "margin"):
                    runs.append([fn, words, idx, probe(as_float[fn], word, words, idx)])
            yield ["fixture_word", name, word, m.route(word), repr(m.margin_tau(word)), runs]
        bifonia._sense_model = lambda: m
        for words, idx in FIXTURE_SENTENCES:
            yield ["fixture_guess", name, words, idx, probe(bifonia.guess_sense, words, idx)]
        for s in ("O autocarro para aqui.", "A sede de água.", "O molho de chaves e a bola."):
            yield ["fixture_diacritics", name, s, probe(bifonia.add_extra_diacritics, s)]
        bifonia._sense_model = default_model
    root = vocab._LOCALE_ROOT
    vocab._LOCALE_ROOT = FIXTURES / "data" / "locale"
    for name in ("badutf", "lines"):
        yield ["fixture_voc", name, probe(vocab.voc, name, "pt-pt")]
    vocab._LOCALE_ROOT = root


# ── output ────────────────────────────────────────────────────────────────────

# Lect tags: each lect and its sub-lects, case, "_", whitespace and script
# subtags, and tags that only look like a lect.
LECT_TAGS = (
    None, "", " ", "\t", "pt-PT", "pt", "pt-BR", "pt-br", "PT-BR", "pt_BR", "pt_br", " pt-BR ",
    "\u3000pt-BR\n", "\xa0pt-UY", "pt-BR-x-rj", "pt-BR-x-sp", "pt-br-x-rj", "pt_BR_x_rj",
    "PT_BR_X_SP", "pT-bR-X-Rj", "pt-Latn-BR", "pt-latn-br", "PT-LATN-UY", "pt_Latn_BR-x-sp",
    "pt-Latn-BR-x-rj", "pt-Latn", "pt-Latn-UY", "pt-Latn-", "pt-Latn-br-", "pt-latnbr",
    "pt-Latn-Latn-BR", "pt-UY", "pt-uy", "pt_UY", "pt-UY-x-mvd", "pt-BRX", "pt-BR_", "pt-BR-",
    "pt-UYX", "pt-BR x", "pt-BR-x", "ptBR", "br", "BR", "pt--BR", "-pt-BR", "pt-BR--",
    "pt-1234-BR", "pt-Lat1-BR", "pt-Latnx-BR", "pt-abc-BR", "pt-Lat_BR", "pt-İatn-BR", "pt-ǅlat-BR", "pt-ﬁat-BR",
    "pt-Ⅻabc-BR", "pt-ǲabc-BR", "pt-ÇÃÕÉ-BR", "pt‐BR", "pt-ΒR", "es-BR", "es-UY", "pt-AO",
    "pt-MZ", "gl", "en-US", "x-pt-BR", "ptbr", "pt-PT-x-BR",
)

LECT_SENTENCES = (
    "O abono de família chegou.", "Eu abono a dívida.", "Abono a dívida.", "O aceno dele foi breve.",
    "Eu aceno ao vizinho.", "O domo da igreja.", "Eu domo o leão.", "O empeno da porta.",
    "Eu empeno a madeira.", "O desempeno da roda.", "Eu desempeno a chapa.", "A soma dá dez.",
    "Ele soma os números.", "Soma tudo.", "SOMA", "abôno", "sóma", "o abono e a soma",
    "O autocarro para em frente ao hospital.", "Eu acordo cedo.",
    "abono, aceno, domo, empeno, desempeno, soma", "",
)


def lect_calls():
    yield ["LECT_OVERRIDES", [[lang, list(rows.items())] for lang, rows in data.LECT_OVERRIDES.items()]]
    spaces = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF and chr(cp).isspace()]
    tags = list(LECT_TAGS) + [f"{c}pt-BR{c}" for c in spaces] + [f"pt-UY{c}x" for c in spaces]
    for tag in tags:
        yield ["lect_overrides", tag, probe(bifonia.lect_overrides, tag), probe(data.lect_overrides, tag)]
        for s in LECT_SENTENCES:
            yield ["lect_diacritics", tag, s, probe(bifonia.add_extra_diacritics, s, tag)]
    # the boundaries of every str.isalpha range, as a four-letter script subtag
    prev = False
    for cp in range(0x110001):
        cur = cp < 0x110000 and not (0xD800 <= cp <= 0xDFFF) and chr(cp).isalpha()
        if cur != prev:
            for x in (cp - 1, cp):
                if 0 <= x < 0x110000 and not (0xD800 <= x <= 0xDFFF):
                    tag = f"pt-{chr(x) * 4}-BR"
                    yield ["lect_alpha", tag, bool(data.lect_overrides(tag))]
        prev = cur


def write(path, lines, header):
    buffer = io.StringIO()
    header = dict(header, count=len(lines))
    buffer.write(dumps(header) + "\n")
    for line in lines:
        buffer.write(line + "\n")
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as handle:
            handle.write(buffer.getvalue().encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-sentences")
    parser.add_argument("--hf-test")
    parser.add_argument("--fuzz", type=int, default=20000)
    args = parser.parse_args()

    corpora = {}
    if args.test_sentences:
        rows = [json.loads(line) for line in
                pathlib.Path(args.test_sentences).read_text(encoding="utf-8").splitlines() if line.strip()]
        corpora["test_sentences"] = [r["sentence"] for r in rows] + [r["sentence"].lower() for r in rows]
        corpora["test_sentences_case"] = [r["sentence"].upper() for r in rows] + \
            [r["sentence"].swapcase() for r in rows]
    if args.hf_test:
        rows = [json.loads(line) for line in
                pathlib.Path(args.hf_test).read_text(encoding="utf-8").splitlines() if line.strip()]
        corpora["hf_test"] = [r["sentence"] for r in rows] + [r["diacritized_sentence"] for r in rows]
    corpora["frames"] = list(frame_sentences())
    corpora["case"] = list(case_sentences())
    corpora["voc_cues"] = list(voc_sentences())
    corpora["fuzz"] = list(fuzz_sentences(args.fuzz, seed=20261102))
    corpora["extras"] = list(extra_sentences())
    corpora["molho_probes"] = list(molho_probe_sentences())
    corpora["molho_frames"] = list(molho_sentences())

    # Detail level per corpus: 2 keeps every field, 1 replaces the feature
    # list by its CRC-32, 0 replaces the whole homograph list by its CRC-32.
    # The C++ test computes the same CRC over the same bytes.
    levels = {"test_sentences": 2, "frames": 1, "extras": 2, "molho_probes": 2, "molho_frames": 1}
    seen, inputs, detail = set(), [], {}
    for name, sentences in corpora.items():
        before = len(inputs)
        for s in sentences:
            if s not in seen:
                seen.add(s)
                inputs.append(s)
                detail[s] = levels.get(name, 0)
        print(f"{name}: {len(sentences)} in, {len(inputs) - before} new", file=sys.stderr)

    versions = {"bifonia": importlib.metadata.version("bifonia")}
    header = {"versions": versions, "python": sys.version.split()[0],
              "unicode": unicodedata.unidata_version}
    out = pathlib.Path(args.out_dir)

    lines = [dumps(sentence_record(s, detail[s])) for s in inputs]
    write(out / "sentences.jsonl.gz", lines, header)
    print(f"sentences: {len(inputs)} in, {len(lines)} out", file=sys.stderr)

    calls = []
    for name, gen in (("tables", table_calls), ("infopedia", infopedia_calls),
                      ("text", text_calls), ("index", index_calls), ("probes", probe_calls),
                      ("bytes", byte_calls), ("byte_sets", byte_set_calls),
                      ("cue_repr", cue_repr_calls), ("lect", lect_calls), ("fixtures", fixture_calls)):
        before = len(calls)
        calls.extend(dumps(c) for c in gen())
        print(f"calls/{name}: {len(calls) - before}", file=sys.stderr)
    write(out / "calls.jsonl.gz", calls, header)
    print(f"calls: {len(calls)} in, {len(calls)} out", file=sys.stderr)

    # Route every word to the model, so the model branch of guess_sense decides.
    forced_inputs = [s for s in inputs if detail[s]] + [s for s in inputs if not detail[s]][::7]
    forced = forced_model()
    bifonia._sense_model = lambda: forced
    forced_lines = [dumps(forced_record(s)) for s in forced_inputs]
    write(out / "forced.jsonl.gz", forced_lines, header)
    print(f"forced: {len(forced_inputs)} in, {len(forced_lines)} out", file=sys.stderr)


if __name__ == "__main__":
    main()
