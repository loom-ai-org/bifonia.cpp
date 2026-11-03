// bifonia/scoring.py: the context scorers for each POS, the POS guess and the
// meaning resolvers for words whose senses share a POS.
#include <algorithm>
#include <initializer_list>

#include "detail.hpp"

namespace bifonia::scoring {

namespace {
using detail::at;
using detail::contains;
using detail::ends_with;

const WordSet& V(const char* name) { return vocab::voc(name); }

// The word lists of scoring.py, bound once.
struct Lists {
    const WordSet& enclitics = V("enclitics");
    const WordSet& after_prep_weak = V("after_prep_weak");
    const WordSet& copula = V("copula");
    const WordSet& intensifiers = V("intensifiers");
    const WordSet& temporal_conj = V("temporal_conj");
    const WordSet& conj_subj = V("conj_subjunctive");
    const WordSet& neg_adv = V("neg_adv");
    const WordSet& passive_aux = V("passive_aux");
    const WordSet& pelo_fixed = V("pelo_fixed");
    const WordSet& pelo_route = V("pelo_route");
    const WordSet& sobre_intens = V("sobre_intensifiers");
    const WordSet& fem_det = V("fem_det");
    const WordSet& masc_det = V("masc_det");
    const WordSet& court_terms = V("court_terms");
    const WordSet& head_verbs = V("head_verbs");
    const WordSet& wood_context = V("wood_context");
    const WordSet& ter = V("possession_verbs");
    const WordSet& trans_verb = V("trans_verbs");
    const WordSet& cut_nouns = V("cut_context");
    const WordSet& colher_ctrl = V("colher_control_verbs");
    const WordSet& para_verb_next = V("para_verb_next");
    const WordSet& adv_bridge = V("adv_bridge");
    const WordSet& optative_adv = V("optative_adv");
    const WordSet& bakeware = V("bakeware");
    const WordSet& mould_cues = V("mould_cues");
    const WordSet& head_state = V("head_state");
    const WordSet& forma_shape_prev = V("forma_shape_prev");
    const WordSet& forma_shape_next = V("forma_shape_next");
    const WordSet& articles = V("articles");
    const WordSet& sing_articles = V("sing_articles");
    const WordSet& de_contractions = V("de_contractions");
    const WordSet& contracted_a = V("contracted_a");
    const WordSet& comparatives = V("comparatives");
    const WordSet& pure_neg = V("pure_negation");
    const WordSet& deferral_adv = V("deferral_adv");
    const WordSet& sobre_adp_quant = V("sobre_adp_quant");
    const WordSet& tola_poss_prev = V("tola_poss_prev");
    const WordSet& person_poss = V("person_possessive");
    const WordSet& material_prep = V("material_prep");
    const WordSet& head_locative = V("head_locative");
    const WordSet& tola_det_prev = V("tola_det_prev");
    const WordSet& molho_noun_prev = V("molho_noun_prev");
    const WordSet& adverbial_de = V("adverbial_de");
    const WordSet& object_pronouns = V("object_pronouns");
    const WordSet& object_det = V("object_determiners");
    const WordSet& a_locutions = V("a_locutions");
    const WordSet& post_verb_adverbs = V("post_verb_adverbs");
    const WordSet& frequency_nouns = V("frequency_nouns");
    const WordSet& time_nouns = V("time_nouns");
    const WordSet& degree_nouns = V("degree_nouns");
    const WordSet& evaluative_nouns = V("evaluative_nouns");
    const WordSet& evaluative_adj = V("evaluative_adjectives");
    const WordSet& duration_modifiers = V("duration_modifiers");
    const WordSet& por_contractions = V("por_contractions");
    const WordSet& bake_purpose = V("bake_purpose");
    const WordSet& first_person_nouns = V("first_person_nouns");
    const WordSet& para_prev_subj = V("para_prev_subj");
    const WordSet& prep_governing = V("prep_governing");
    const WordSet& clitics_all = V("clitics_all");
    const WordSet& excl_det = V("exclamative_det");
    const WordSet& locative_contractions = V("locative_contractions");
    const WordSet& func_extra = V("function_words");
    const WordSet& prenominal_adj = V("prenominal_adj");
    const WordSet& complement_preps = V("complement_preps");
    const WordSet& coordination = V("coordination");
    const WordSet& around_prep = V("around_prep");
    const WordSet& sede_seat = V("sede_seat_cues");
    const WordSet& sede_thirst = V("sede_thirst_cues");
    const WordSet& bundle_things = V("bundle_things");
    const WordSet& bundle_ambig = V("bundle_ambiguous");
    const WordSet& bundle_verbs = V("bundle_verbs");
    const WordSet& contracted_det = V("contracted_det");
    const data::Tables& t = data::tables();
};

const Lists& L() {
    static const Lists lists;
    return lists;
}

std::string S(const Words& words, long i) { return text::strip_punct(at(words, i)); }

long size(const Words& words) { return static_cast<long>(words.size()); }

std::string prev(const Words& words, long idx) { return idx > 0 ? S(words, idx - 1) : ""; }
std::string next(const Words& words, long idx) { return idx + 1 < size(words) ? S(words, idx + 1) : ""; }
std::string prev2(const Words& words, long idx) { return idx > 1 ? S(words, idx - 2) : ""; }

bool in(const std::string& w, std::initializer_list<const WordSet*> sets) {
    for (const WordSet* s : sets) {
        if (contains(*s, w)) return true;
    }
    return false;
}

bool ends_with_any(const std::string& w, std::initializer_list<const char*> suffixes) {
    for (const char* s : suffixes) {
        if (ends_with(w, s)) return true;
    }
    return false;
}

bool is_infinitive(const std::string& word) {
    return ends_with_any(detail::rstrip_chars(word, ".,;:!?"), {"ar", "er", "ir"});
}

// A digit string, optionally negative, or a cardinal number word.
bool is_numeric(const std::string& word) {
    const std::string digits = detail::starts_with(word, "-") ? word.substr(1) : word;
    return detail::py_isdecimal(digits) || contains(L().t.NUMERIC, word);
}

// Python's `word[-1:] in chars`, which is also true for an empty word.
bool ends_in(const std::string& word, std::string_view chars) {
    return word.empty() || detail::last_in(word, chars);
}

// words[max(0, idx - k)] stripped, for k = 1..count where idx - k >= 0.
std::vector<std::string> left(const Words& words, long idx, long count) {
    std::vector<std::string> out;
    for (long k = 1; k <= count; ++k) {
        if (idx - k >= 0) out.push_back(S(words, std::max(0L, idx - k)));
    }
    return out;
}

// words[idx + k] stripped, for k = 1..count where idx + k < len(words).
std::vector<std::string> right(const Words& words, long idx, long count) {
    std::vector<std::string> out;
    for (long k = 1; k <= count; ++k) {
        if (idx + k < size(words)) out.push_back(S(words, idx + k));
    }
    return out;
}

bool any_in(const std::vector<std::string>& ws, const WordSet& set) {
    return std::any_of(ws.begin(), ws.end(), [&](const std::string& w) { return contains(set, w); });
}

// w[-n] is not one of vowels (code points; w must be long enough).
bool cp_from_end_not_in(const std::string& w, size_t n, std::u32string_view vowels) {
    const auto u = detail::from_utf8(w);
    return vowels.find(u[u.size() - n]) == std::u32string_view::npos;
}

std::string resolve_sede(const Words& words, long idx) {
    const Lists& l = L();
    const std::string prev_word = prev(words, idx);
    const std::string next_word = next(words, idx);
    const std::string next2 = idx + 2 < size(words) ? S(words, idx + 2) : "";

    int seat = 0, thirst = 0;
    if (next_word == "da" || next_word == "do" || next_word == "das" || next_word == "dos") seat += 3;
    if (next_word == "de") {
        if (contains(l.t.DET, next2)) {
            seat += 3;
        } else if (contains(l.sede_seat, next2)) {
            seat += 4;
        } else if (contains(l.sede_thirst, next2) ||
                   ends_with_any(next2, {"ção", "são", "mento", "dade", "ência", "ância", "ismo", "tude"})) {
            thirst += 4;
        } else {
            seat += 2;
        }
    }
    if (prev_word == "de") thirst += 3;
    if (prev_word == "na" || prev_word == "à" || prev_word == "numa" || prev_word == "pela" ||
        prev_word == "duma") {
        seat += 2;
    }
    if (next_word == "no" || next_word == "na" || next_word == "nos" || next_word == "nas" ||
        next_word == "em") {
        seat += 3;
    }
    const long lo = std::max(0L, idx - 3), hi = std::min(size(words), idx + 4);
    for (long i = lo; i < hi; ++i) {
        if (i == idx) continue;
        const std::string w = S(words, i);
        seat += contains(l.sede_seat, w);
        thirst += contains(l.sede_thirst, w);
    }
    if (seat > thirst) return "seat";
    if (thirst > seat) return "thirst";
    const std::string* d = l.t.DEFAULT_SENSE.find("sede");
    return d ? *d : "thirst";
}

std::string resolve_molho(const Words& words, long idx) {
    const Lists& l = L();
    const std::string next_word = next(words, idx);
    const std::string m2 = idx + 2 < size(words) ? S(words, idx + 2) : "";
    const auto left4 = left(words, idx, 4);
    if (next_word == "de" && contains(l.bundle_things, m2)) return "bundle";
    if (next_word == "de" && contains(l.bundle_ambig, m2) && any_in(left4, l.bundle_verbs)) return "bundle";
    return "sauce";
}

// A clause-initial "molho" followed by its direct object, past adverbs and
// fixed locutions, is the verb molhar.
bool molho_takes_object(const Words& words, long idx) {
    const Lists& l = L();
    auto adverb = [&](const std::string& w) {
        return w == "mais" ||
               in(w, {&l.adv_bridge, &l.neg_adv, &l.post_verb_adverbs, &l.deferral_adv, &l.intensifiers});
    };
    if (detail::starts_with(next(words, idx), "-")) return true;
    long j = idx + 1;
    while (j < size(words)) {
        const std::string tok = S(words, j);
        const std::string after = j + 1 < size(words) ? S(words, j + 1) : "";
        const std::string after2 = j + 2 < size(words) ? S(words, j + 2) : "";
        const bool um = tok == "um" || tok == "uma";
        if (um && (contains(l.evaluative_nouns, after) ||
                   (contains(l.evaluative_adj, after) && contains(l.evaluative_nouns, after2)))) {
            return false;
        }
        if (um && contains(l.degree_nouns, after) && after2 == "de") return true;
        if (detail::starts_with(after, "-") && in(tok, {&l.object_pronouns, &l.object_det})) return false;
        const bool todo = tok == "todo" || tok == "toda" || tok == "todos" || tok == "todas" || um;
        if (todo && (contains(l.time_nouns, after) ||
                     (contains(l.articles, after) && contains(l.time_nouns, after2)))) {
            j += contains(l.time_nouns, after) ? 1 : 2;
            if (j + 1 < size(words) && contains(l.duration_modifiers, S(words, j + 1))) ++j;
        } else if ((tok == "de" && contains(l.adverbial_de, after)) ||
                   ((tok == "a" || tok == "à") && contains(l.a_locutions, after)) ||
                   (in(tok, {&l.t.NUMERIC, &l.t.QUANT}) && contains(l.frequency_nouns, after)) ||
                   (um && contains(l.degree_nouns, after))) {
            ++j;
        } else if (in(tok, {&l.object_pronouns, &l.object_det})) {
            return true;
        } else if (!adverb(tok)) {
            return false;
        }
        if (ends_in(at(words, j), ".!?;:,")) return false;
        ++j;
    }
    return false;
}

std::string resolve_by_cues(const std::string& word, const Words& words, long idx) {
    const cues::CueRule& rule = cues::SENSE_CUES().at(word);
    std::string best = rule.default_sense;
    int best_score = 0;
    for (const auto& cue : rule.cues) {
        if (cue.requires_prev && prev(words, idx) != *cue.requires_prev) continue;
        const int score = text::cue_score(words, idx, detail::folded_voc(cue.voc));
        if (score > best_score) {
            best = cue.sense;
            best_score = score;
        }
    }
    return best;
}
}  // namespace

int score_adp(const Words& words, long idx) {
    const Lists& l = L();
    const data::Tables& t = l.t;
    const std::string& word = at(words, idx);
    const std::string prev_word = prev(words, idx);
    const std::string next_word = next(words, idx);

    if (contains(t.NEVER_AFTER_PREP, next_word)) return 0;

    int score = 0;
    if (contains(t.BEFORE_PREP, prev_word)) score += 5;
    if (contains(t.AFTER_PREP, next_word)) score += contains(l.after_prep_weak, next_word) ? 2 : 5;
    if (is_numeric(next_word)) score += 5;
    if (is_infinitive(next_word)) score += 2;

    if (word == "para" && contains(l.pure_neg, next_word)) score += 4;

    if (word == "para") {
        if (ends_with_any(next_word, {"ção", "são", "gem", "ura", "ência", "ância", "mento", "ismo", "ise"})) {
            score += 4;
        }
        if (contains(l.deferral_adv, next_word)) score += 3;
        const std::string para_prev2 = idx >= 2 ? S(words, idx - 2) : "";
        if (contains(l.copula, para_prev2)) score += 4;
    }

    if (word == "pelo") {
        if (contains(l.pelo_fixed, next_word)) score += 6;
        if (contains(l.pelo_route, next_word)) score += 5;
        static const char* const kPpt[] = {"ado", "ido", "ada", "ida", "ados", "idos", "adas", "idas"};
        auto is_ppt = [](const std::string& w) {
            for (const char* sfx : kPpt) {
                const size_t n = std::char_traits<char>::length(sfx);
                if (ends_with(w, sfx) && detail::cp_len(w) > n &&
                    cp_from_end_not_in(w, n + 1, U"aeiouáéíóúâêîôûãõ")) {
                    return true;
                }
            }
            return false;
        };
        const auto left3 = left(words, idx, 3);
        if (std::any_of(left3.begin(), left3.end(), is_ppt)) score += 5;
        if (contains(l.articles, next_word)) {
            score -= 8;
        } else if (idx > 0) {
            score += 3;
        }
    }

    if (word == "sobre") {
        if (contains(t.SOBRE_GOV, prev_word)) score += 6;
        if (contains(l.sobre_adp_quant, prev_word)) score += 5;
        if (in(next_word, {&t.DET, &t.PRON})) score += 3;
        if (idx > 0 && !in(prev_word, {&t.PRON, &l.neg_adv, &l.temporal_conj})) score += 2;
    }
    return score;
}

int score_noun(const Words& words, long idx) {
    const Lists& l = L();
    const data::Tables& t = l.t;
    const std::string prev_word = prev(words, idx);
    const std::string next_word = next(words, idx);
    const std::string& word = at(words, idx);
    int score = 0;

    if (in(prev_word, {&t.DET, &t.QUANT}) && !(word == "sobre" && contains(l.sobre_intens, prev_word)) &&
        word != "tola") {
        score += 5;
        if (word == "sobre") score += 7;
    }
    if (word == "posto" && contains(l.passive_aux, prev_word)) score += 8;
    if (contains(l.de_contractions, next_word)) score += 3;
    if (word == "torno" && contains(l.around_prep, prev_word)) score += 6;
    if (word == "troco" && prev_word == "em") score += 6;
    if (contains(l.prenominal_adj, prev_word)) score += 5;
    if (contains(l.complement_preps, prev_word) && word != "colher") score += 3;
    if (contains(l.coordination, prev_word)) score += 3;
    if (contains(l.trans_verb, prev_word)) score += 3;
    if (word == "pelo" && contains(l.ter, prev_word)) score += 6;
    if (word == "pelo" && contains(l.articles, prev_word)) score += 6;
    if (word == "pelo" && (next_word == "do" || next_word == "da")) score += 5;
    if (word == "pelo" && prev_word == "de") score += 6;
    if (word == "corte" && contains(l.fem_det, prev_word)) score += 7;
    if (word == "corte" && contains(l.court_terms, next_word)) score += 3;

    if (word == "tola") {
        const auto left3 = left(words, idx, 3);
        if (any_in(left3, l.head_verbs)) score += 6;
        if (any_in(right(words, idx, 2), l.head_state)) score += 6;
        if (contains(l.tola_poss_prev, prev_word) && contains(l.person_poss, next_word)) score += 5;
        if (contains(l.material_prep, prev_word)) score += 4;
        if (contains(l.head_locative, prev_word)) score += 5;
        const std::string prev2_tola = idx >= 2 ? S(words, idx - 2) : "";
        if (prev2_tola == "com" && contains(l.sing_articles, prev_word)) score += 5;
        if (contains(l.contracted_a, next_word) && contains(l.tola_det_prev, prev_word)) score += 4;
        auto both = left3;
        const auto right3 = right(words, idx, 3);
        both.insert(both.end(), right3.begin(), right3.end());
        if (any_in(both, l.wood_context)) score += 6;
    }

    if (word == "forma") {
        const std::string next2_f = idx + 2 < size(words) ? S(words, idx + 2) : "";
        std::vector<std::string> toks_f;
        for (const auto& w : words) toks_f.push_back(text::strip_punct(w));
        const bool shape = contains(l.forma_shape_prev, prev_word) || next_word == "como" ||
                           (next_word == "de" && is_infinitive(next2_f)) ||
                           contains(l.forma_shape_next, next2_f);
        const bool purpose =
            next_word == "para" &&
            (contains(l.bake_purpose, next2_f) || contains(l.bakeware, next2_f) ||
             (contains(l.sing_articles, next2_f) &&
              contains(l.bakeware, text::strip_punct(idx + 3 < size(words) ? at(words, idx + 3) : ""))));
        if (shape) {
            score = 0;
        } else if (any_in(toks_f, l.mould_cues) || (next_word == "de" && contains(l.bakeware, next2_f)) ||
                   purpose) {
            score += 6;
        } else {
            score = 0;
        }
    }

    if (word == "molho") {
        const bool no_object = !in(next_word, {&l.articles, &t.QUANT}) && !detail::starts_with(next_word, "-");
        const bool relative = in(prev2(words, idx), {&t.DET, &l.por_contractions});
        if ((prev_word == "qual" || prev_word == "quais") && !relative) {
            score += 12;
        } else if (contains(l.molho_noun_prev, prev_word) && no_object) {
            score += 8;
        }
        if (idx == 0 || ends_in(at(words, idx - 1), ",;:")) {
            if (!molho_takes_object(words, idx)) score += 8;
        } else if (next_word == "de") {
            const std::string after_de = idx + 2 < size(words) ? S(words, idx + 2) : "";
            if (!contains(l.adverbial_de, after_de)) score += 4;
        }
    }
    return score;
}

int score_verb(const Words& words, long idx) {
    const Lists& l = L();
    const data::Tables& t = l.t;
    const std::string& word = at(words, idx);
    const std::string prev_word = prev(words, idx);
    const std::string next_word = next(words, idx);
    const std::string prev2_word = prev2(words, idx);
    const long n = size(words);

    int score = 0;
    if (idx == 0) score += 3;

    if (contains(t.PRON, prev_word)) {
        score += 5;
        if (idx == 1) score += 2;
    }

    if (idx > 0 && !in(prev_word, {&t.DET, &t.QUANT}) && !in(prev2_word, {&t.DET, &t.QUANT}) &&
        !contains(l.trans_verb, prev_word) && !contains(l.complement_preps, prev_word)) {
        if (contains(l.first_person_nouns, word)) score += 2;
    }

    const std::string next2_v = idx + 2 < n ? S(words, idx + 2) : "";
    if (contains(l.first_person_nouns, word) &&
        (contains(l.complement_preps, next_word) || next_word == "para" || next_word == "a" ||
         next_word == "ao" || next_word == "à") &&
        contains(t.PRON, next2_v) &&
        !in(prev_word, {&t.DET, &t.QUANT, &l.complement_preps, &l.trans_verb})) {
        score += 5;
    }

    if (word != "para" && detail::starts_with(next_word, "-") && contains(l.enclitics, next_word.substr(1))) {
        score += 4;
    }

    const std::string next_next = idx + 2 < n ? at(words, idx + 2) : "";
    const bool posto_inf = word == "posto" && next_word == "a" && is_infinitive(next_next);
    static const WordSet kPostposedQuant = {"todo", "toda", "todos", "todas", "mesmo", "mesma", "inteiro", "inteira"};
    if (word != "para" && in(next_word, {&t.DET, &t.QUANT}) && !contains(l.contracted_det, next_word) &&
        !contains(kPostposedQuant, next_word) && !posto_inf) {
        score += 3;
    } else if (idx == 0 && contains(l.de_contractions, next_word)) {
        score += 1;
    }

    if (word == "corte" && contains(l.masc_det, prev_word)) score += 10;
    if (word == "corte") {
        auto around = left(words, idx, 2);
        const auto right2 = right(words, idx, 2);
        around.insert(around.end(), right2.begin(), right2.end());
        if (any_in(around, l.cut_nouns)) score += 4;
    }

    if (contains(l.temporal_conj, prev_word)) score += 2;
    if (contains(l.conj_subj, prev_word)) score += 4;
    if (contains(l.neg_adv, prev_word)) score += 4;
    if (contains(l.passive_aux, prev_word)) score += 4;
    if (word == "para" && is_infinitive(next_word)) score -= 5;

    if (word == "para") {
        if (any_in(left(words, idx, 4), t.STOPPABLE_THINGS)) score += 4;
        if (contains(t.STOPPABLE_THINGS, next_word)) score += 3;
        const std::string next2 = idx + 2 < n ? S(words, idx + 2) : "";
        if (contains(l.articles, next_word) && contains(t.STOPPABLE_THINGS, next2)) score += 3;
        if (next_word == "a" && next2 == "meio") score += 4;
    }

    const std::string prev_raw_v = idx > 0 ? at(words, idx - 1) : "";
    if (is_infinitive(prev_word) && !detail::last_in(prev_raw_v, ".,;:!?")) score -= 5;

    if (is_infinitive(word) && !in(prev_word, {&t.DET, &t.QUANT})) score += 5;
    if (word == "colher" && prev_word == "a" && in(next_word, {&t.DET, &t.QUANT}) &&
        !contains(l.contracted_det, next_word)) {
        score += 6;
    }
    const std::string prev3 = idx >= 3 ? S(words, idx - 3) : "";
    const std::string prev4 = idx >= 4 ? S(words, idx - 4) : "";
    const bool colher_ctrl_ok = !is_infinitive(prev2_word);
    if (word == "colher" && prev_word == "a" && colher_ctrl_ok &&
        (contains(l.colher_ctrl, prev2_word) || contains(l.colher_ctrl, prev3) || contains(l.colher_ctrl, prev4))) {
        score += 6;
    }

    if (contains(t.DET, prev2_word) && !contains(l.locative_contractions, prev2_word)) score += 3;

    if (word == "para" && contains(l.para_prev_subj, prev_word)) score += 3;

    if (word == "para") {
        if (contains(l.para_verb_next, next_word)) score += 5;
        const std::string next_next_v = idx + 2 < n ? at(words, idx + 2) : "";
        if (next_word == "de" && is_infinitive(next_next_v)) score += 5;
        if (contains(l.temporal_conj, next_word)) score += 3;
    }

    if (ends_with(next_word, "mente")) score += 3;

    const std::string next2 = idx + 2 < n ? S(words, idx + 2) : "";
    if (contains(l.adv_bridge, next_word) && in(next2, {&t.DET, &t.QUANT})) score += 1;

    if (word == "gozo" && next_word == "de" && !in(prev_word, {&t.DET, &t.QUANT, &l.prep_governing})) score += 5;

    if (word == "sobre" && prev_word == "sempre" && in(next_word, {&t.DET, &t.QUANT})) score += 8;

    if (word == "sobre") {
        if (ends_with(prev_word, "mente")) score += 4;
        const auto left3s = left(words, idx, 3);
        if (std::find(left3s.begin(), left3s.end(), "depois") != left3s.end() ||
            std::find(left3s.begin(), left3s.end(), "restam") != left3s.end()) {
            score += 3;
        }
        if (contains(l.optative_adv, prev_word)) score += 6;
    }
    return score;
}

int score_adj(const Words& words, long idx) {
    const Lists& l = L();
    const data::Tables& t = l.t;
    const std::string prev_word = prev(words, idx);
    const std::string prev2_word = prev2(words, idx);
    const std::string next_word = next(words, idx);

    int score = 0;
    if (contains(l.copula, prev_word)) score += 5;
    if (contains(l.comparatives, prev_word)) score += 4;
    if (contains(l.intensifiers, prev_word)) score += 2;
    if (ends_with(prev_word, "mente")) score += 4;
    if (contains(l.copula, prev2_word) && ends_with(prev_word, "mente")) score += 3;

    const std::string prev_raw_adj = idx > 0 ? at(words, idx - 1) : "";
    const bool prev_across_boundary = prev_word.empty() || detail::last_in(prev_raw_adj, ".,;:!?");
    if (in(prev2_word, {&t.DET, &t.QUANT, &l.excl_det}) && !in(prev_word, {&t.DET, &t.QUANT, &l.excl_det}) &&
        !prev_across_boundary) {
        score += 4;
    }

    if (in(prev_word, {&t.DET, &t.QUANT, &t.PRON})) score += 3;

    const std::string prev_raw = idx > 0 ? at(words, idx - 1) : "";
    bool prev_looks_verb = ends_with_any(prev_word, {"ou", "eu", "iu", "ei", "ava", "ara", "era"});
    if (!prev_looks_verb && ends_with(prev_word, "ia") && detail::cp_len(prev_word) >= 4 &&
        cp_from_end_not_in(prev_word, 3, U"aeiouáéíóúâêîôûãõàèìòùäëïöü")) {
        prev_looks_verb = true;
    }
    if (!prev_looks_verb) {
        const size_t dash = prev_word.rfind('-');
        if (dash != std::string::npos && contains(l.clitics_all, prev_word.substr(dash + 1))) prev_looks_verb = true;
    }
    if (!prev_word.empty() && detail::cp_len(prev_word) > 2 && !detail::last_in(prev_raw, ".,;:!?") &&
        !in(prev_word, {&t.DET, &t.QUANT, &t.PRON, &t.AUX_VERBS, &l.func_extra}) &&
        !in(next_word, {&t.DET, &t.QUANT}) && !prev_looks_verb) {
        score += 3;
    }
    return score;
}

std::string guess_pos(const Words& words, long idx, bool proper) {
    const Lists& l = L();
    const data::Tables& t = l.t;
    const std::string& word = at(words, idx);
    static const Dict<int> kNoBias;
    const Dict<int>* found = t.BASE_SCORE.find(word);
    const Dict<int>& bias = found ? *found : kNoBias;
    auto b = [&](const char* pos) {
        const int* v = bias.find(pos);
        return v ? *v : 0;
    };

    Dict<int> scores;
    if (t.ADP_IPA.contains(word)) scores["ADP"] = score_adp(words, idx) + b("ADP");
    if (t.NOUNS_IPA.contains(word)) scores["NOUN"] = score_noun(words, idx) + b("NOUN");
    if (t.VERBS_IPA.contains(word)) scores["VERB"] = score_verb(words, idx) + b("VERB");
    if (t.ADJ_IPA.contains(word)) scores["ADJ"] = score_adj(words, idx) + b("ADJ");

    if (proper) {
        if (int* noun = scores.find("NOUN")) {
            const std::string nxt = next(words, idx);
            if (!(detail::starts_with(nxt, "-") && contains(l.enclitics, nxt.substr(1)))) *noun += 12;
        }
    }

    if (scores.empty()) throw ValueError("max() arg is an empty sequence");
    int best = scores.items()[0].second;
    for (const auto& item : scores.items()) best = std::max(best, item.second);
    if (best <= 0) {
        const std::string* d = t.DEFAULT_POS.find(word);
        return d ? *d : "NOUN";
    }
    std::vector<std::string> winners;
    for (const auto& [pos, s] : scores.items()) {
        if (s == best) winners.push_back(pos);
    }
    if (winners.size() == 1) return winners[0];
    const std::string* d = t.DEFAULT_POS.find(word);
    return d ? *d : winners[0];
}

std::optional<std::string> resolve_sense(const std::string& word, const Words& words, long idx,
                                         const std::string& pos) {
    const data::Tables& t = L().t;
    const Dict<std::vector<std::string>>* pmap = t.POS_SENSES.find(word);
    const std::vector<std::string>* senses = pmap ? pmap->find(pos) : nullptr;
    if (senses == nullptr && pmap != nullptr) {
        const std::string* d = t.DEFAULT_POS.find(word);
        senses = pmap->find(d ? *d : "");
    }
    if (senses == nullptr) {
        if (pmap == nullptr || pmap->empty()) return std::nullopt;  // [None]
        senses = &pmap->items()[0].second;
    }
    if (senses->size() == 1) return (*senses)[0];
    if (cues::SENSE_CUES().contains(word)) return resolve_by_cues(word, words, idx);
    if (word == "sede") return resolve_sede(words, idx);
    if (word == "molho") return resolve_molho(words, idx);
    if (word == "tola") return std::string("head");
    return (*senses)[0];
}

}  // namespace bifonia::scoring
