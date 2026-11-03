# Parity with the Python package

This library is held at byte parity with the Python `bifonia` release named in
`kPythonReference` (`include/bifonia/bifonia.hpp`). The test `parity` checks that
claim. It recomputes every line of the three files in `tests/goldens/` and
compares the bytes.

## Public surface

The Python package has 11 modules: the package itself and `corpus`, `cues`,
`data`, `features`, `infopedia`, `model`, `scoring`, `text`, `version` and
`vocab`. Together they define 67 public names, counting each re-export once.
The port has 63 of them. The 4 names of `corpus` are not ported, for the reason
given below.

| Python | C++ |
| --- | --- |
| `bifonia.tokenize` | `bifonia::tokenize` |
| `bifonia.is_ambiguous` | `bifonia::is_ambiguous` |
| `bifonia.disambiguate` | `bifonia::disambiguate` |
| `bifonia.guess_sense` | `bifonia::guess_sense` (empty `std::optional` where Python returns `None`) |
| `bifonia.guess_pos` | `bifonia::guess_pos` |
| `bifonia.proper_flags` | `bifonia::proper_flags` |
| `bifonia.add_extra_diacritics` | `bifonia::add_extra_diacritics` (an empty `std::optional` for `lang` where Python passes `None`) |
| `HOMOGRAPHS`, `AMBIGUOUS_WORDS`, `POS_SENSES`, `SENSE_POS` (also in `data`) | `bifonia::HOMOGRAPHS()` and the others, and the members of `data::Tables` |
| `lect_overrides` (also in `data`) | `bifonia::lect_overrides`, `data::lect_overrides` |
| `data.LECT_OVERRIDES` | `data::Tables::LECT_OVERRIDES` |
| `data.SENSE_POSES`, `POS_AMBIGUOUS`, `VERBS_IPA`, `NOUNS_IPA`, `ADP_IPA`, `ADJ_IPA`, `DEFAULT_SENSE`, `DEFAULT_POS`, `BASE_SCORE` | members of `data::Tables`, from `data::tables()` |
| `data.STOPPABLE_THINGS`, `DET`, `QUANT`, `PRON`, `AUX_VERBS`, `NUMERIC`, `SOBRE_GOV`, `NEVER_AFTER_PREP`, `PREP`, `BEFORE_PREP`, `AFTER_PREP` | members of `data::Tables` |
| `scoring.score_adp`, `score_noun`, `score_verb`, `score_adj` | `scoring::score_adp` and the others |
| `scoring.guess_pos`, `scoring.resolve_sense` | `scoring::guess_pos`, `scoring::resolve_sense` |
| `cues.Cue`, `cues.CueRule` (`default` is `default_sense` in C++) | `cues::Cue`, `cues::CueRule` |
| `Cue.__repr__`, `CueRule.__repr__`, `__eq__` | `cues::repr`, `operator==` |
| `cues.SENSE_CUES`, `cues.FEATURE_CUES` | `cues::SENSE_CUES()`, `cues::FEATURE_CUES()` |
| `text.strip_punct`, `fold`, `folded`, `cue_score` | `text::strip_punct` and the others |
| `features.STRUCTURAL_VOCS`, `load_structural_vocs`, `extract_features` | `features::STRUCTURAL_VOCS()` and the others |
| `model.NB_PATH`, `model.PERCEPTRON_PATH` | `model::NB_PATH()`, `model::PERCEPTRON_PATH()` |
| `model.SenseModel.load`, `has`, `route`, `margin_tau`, `scores`, `predict`, `margin` | the members of `model::SenseModel` with the same names |
| `SenseModel.version`, `model_type`, `config`, `lang`, `words` | the fields of the same names, with each `config` value held as its JSON text |
| `vocab.voc`, `vocab.DEFAULT_LANG` | `vocab::voc`, `vocab::DEFAULT_LANG` |
| `infopedia.load`, `readings`, `ipa` | `infopedia::load`, `readings`, `ipa` |
| `version.VERSION_*`, `bifonia.__version__` | `version::VERSION_*`, `kPythonReference` |

Two private Python names have a C++ counterpart because the goldens use them.
`bifonia._DIACRITIZED` is `bifonia::DIACRITIZED()`, read-only. Assigning
`bifonia._sense_model` is `bifonia::set_sense_model`.

A Python dict maps to `bifonia::Dict`, which keeps insertion order. The order
decides the result in several places: score ties in `guess_pos` (ADP, NOUN, VERB,
ADJ), the fallback sense in `resolve_sense`, the summation order of the model
scores, and the last-wins reverse maps of the diacritic table. A Python set or
frozenset maps to `std::unordered_set`. No result depends on set order.

The tie order of `guess_pos` cannot show in a result. On a tie, `guess_pos`
returns `DEFAULT_POS[word]`, and `DEFAULT_POS` has an entry for each homograph.
Only a homograph gets POS scores, so the first winner of a tie is never used.
A mutant that changes this order passes all goldens for that reason.

### Not ported

| Python | Reason |
| --- | --- |
| `corpus.CORPUS`, `corpus.IPA`, `corpus.iter_records`, `corpus.stats` | They only expose the 22.7 MB training corpus `corpus.jsonl`, which no inference path reads. The corpus is not shipped. |
| `data/meaning_inventory.csv` | No module reads it. |
| `SenseModel(blob)` | The C++ model is built by `SenseModel::load`, or by copying one and setting its public fields. |
| The tuple protocol of `Cue` and `CueRule` (`__len__`, `__iter__`, `__getitem__`, `_replace`, `_asdict`, `_make`, `_fields`) | They are plain structs in C++. |

The package has no command-line interface, so this library ships none.

## Differences from bifonia 0.2.0a4

The table lists each public name whose behaviour differs between bifonia
0.2.0a4 and the reference release. The port follows the reference release.

| Python | Difference |
| --- | --- |
| `add_extra_diacritics` | Takes `lang`, the lect tag. For a lect with rows in `lect_overrides.csv`, the override form replaces the marked form of the word in every sense, and an empty form leaves the word unmarked. |
| `lect_overrides`, `data.lect_overrides`, `data.LECT_OVERRIDES` | Added. The rows are for `pt-BR` and `pt-UY`. |
| `tokenize` | Lowercases each token after the split. A code point whose lowercase is two code points (`İ`) no longer splits a token, so the tokens align with `proper_flags`. |
| `proper_flags` | Reads the tokens `n.` and `º` as `nº`. The next token `no` or `n.º` no longer marks the first token. |
| `guess_sense`, `guess_pos` | A diacritized form that marks several senses (`chôco`: addled and cuttlefish) narrows the choice, and the context picks among those senses. |
| `scoring.score_noun` | Scores `molho` as a noun after an interrogative or a verb of lack or surplus, at the start of a clause with no direct object, and before `de` and a noun. |
| `scoring.score_adp` | A number is a string of decimal digits with an optional leading `-` (`str.isdecimal`), or a number word. It was any string that `float()` accepts. |
| `infopedia.ipa` | `pos` is a UD tag. A reading with that tag alone wins over a reading whose `|`-joined tags include it. |
| `data.POS_SENSES`, `data.SENSE_POSES`, `data.VERBS_IPA`, `infopedia.load` | The verb molhar (`eu molho`) has the open vowel of the `bundle` sense, not the closed vowel of `sauce`. |
| `vocab.voc` | 13 word lists added for `molho`. |
| `version.VERSION_*` | 0.3.7a1. |

## The sense models

Both shipped models route every one of their 131 words to the rules
(`route == "rules"`). The model therefore never decides in default use: the
check in `guess_sense` loads the perceptron and falls through to the rules. The
port still carries the whole model path. The `forced` goldens copy the
perceptron, route each word to the model with a `margin_tau` of 0, 0.75 or 1.5,
and install the copy, so that `guess_sense` takes the model branch. Every
sentence line also records both models' scores, prediction and margin for each
homograph.

The model files are read by a small JSON reader made for their schema, so the
library has no JSON dependency. It accepts and rejects the same text as Python's
`json` module: the strict number grammar, `NaN`, `Infinity` and `-Infinity`, and
no raw control characters in a string. Numbers are read as doubles with
`std::from_chars`, which rounds as Python's `float()` does. A number past the
double range reads as an infinity, and a number below it reads as a zero, with
the sign kept. An integer of more than 4300 digits raises `ValueError`, as
Python's `int()` does, and the integer `-0` reads as 0. `SenseModel::version`
and each `config` value keep their JSON text, because Python keeps any JSON
value there. The text is the source text. It equals Python's `json.dumps` of the
value only when the source is in that form: the source `1E2` stays `1E2`, and
`json.dumps` gives `100.0`.

A score is a sum of products. Python rounds each product and each sum. A fused
multiply-add rounds once, and the last bit of a score then differs. The library
target therefore compiles with `-ffp-contract=off` on GCC and Clang. A CI cell
builds for `x86-64-v3`, which has FMA. Without the flag, that build gives 81
mismatches.

x87 arithmetic (a 32-bit x86 build) keeps intermediates at 80 bits. Each product
goes into a named double before the sum, and the library target compiles with
`-fexcess-precision=standard` on GCC 13 and later, so that each assignment
rounds to double. The flag is on the target, so it overrides the
`-fexcess-precision=fast` of GCC's GNU mode in a parent project. The CI cell
`x87` builds with `-m32 -fexcess-precision=fast` in the project flags and runs
the goldens.

`margin` sorts the scores as CPython 3.12 sorts a list with `reverse=True`. The
order differs from a plain sort only when a score is NaN. A score can be NaN
without a NaN literal: `1e400` plus `-1e400` gives NaN. CPython 3.13 changed how
`list.sort` finds runs, so Python 3.13 and later order NaN scores in another
way. The port follows Python 3.12, the reference version.

## Known divergences on valid model files

Each case below needs a model file that the package does not ship. The file is
valid JSON, and Python's `json` module loads it.

| Input | Python | C++ |
| --- | --- | --- |
| An integer above 2^53 in magnitude | Keeps the exact `int`. Biases 9007199254740992 and 9007199254740993 predict the second sense. Biases 100000000000000001 and 100000000000000000 give a margin of 1 | Reads the nearest double. The same biases predict the first sense and give a margin of 0.0 |
| An integer past the double range | Raises `OverflowError` when a score or a margin mixes it with a float. When only integers meet, compares them exactly: with biases 10^400 − 1 and 10^400, `predict` gives the sense with the larger bias, and the margin is 1 | Reads ±inf. When only such integers meet, the scores tie at inf: `predict` gives the sense name that sorts first, the margin is inf − inf = NaN, and `guess_sense` falls back to the rules |
| A value that stays an integer: an integer `margin_tau`, or a score or margin from an integer bias with no matching weight | Prints without `.0` | Prints with `.0` |
| A NaN score in a word with 64 senses or more | Merges sorted runs | Sorts by insertion only |
| A `\u` escape of a lone surrogate | Keeps the surrogate | Reads U+FFFD |

## Known divergences on malformed input

Each case below needs a model or data file that the package does not ship, or
text that is not UTF-8. The model files are valid JSON but not valid models.

| Input | Python | C++ |
| --- | --- | --- |
| A model value of the wrong JSON type, such as `"route": null`, `"route": 1` or `"margin_tau": "0"` | Loads the file. `route` returns the value, and a string `margin_tau` raises `TypeError` when the model decides | Raises `JSONDecodeError` at load |
| A model file that is not an object, or a `config` that is not an object | Raises `AttributeError` at load | Raises `JSONDecodeError` at load |
| `config.lang` that is not a string | Raises `AttributeError` when the model scores | Uses `pt-pt` |
| A word entry without `senses`, `bias` or `weights` | Raises `KeyError` when the model scores | Uses an empty list or map |
| `Cue` or `CueRule` equality on ill-formed UTF-8 | Compares the decoded strings | Compares the bytes |

## Unicode and strings

Strings are UTF-8. An ill-formed byte sequence reads as Python's
`bytes.decode("utf-8", "replace")` reads it: one U+FFFD for each maximal
subpart. The tables in `src/unicode_tables.inc` come from Python itself
(`tools/gen_unicode_tables.py`) and record the Python and Unicode versions. They
give `str.lower` (with the Greek final sigma), `str.upper`, `str.isspace`, the
`\w` class of `re`, `str.isupper`, `str.isdigit`, `str.isalpha` and
`str.isprintable` on one code point, the decimal digits of `str.isdecimal`, the
NFD decompositions with the canonical combining classes, and category Mn. Each
regular expression is written out as code.

A set lookup compares a string with ill-formed bytes as the string Python
decodes it to. A data file that is not UTF-8 raises `UnicodeDecodeError`, as
Python's `open(path, encoding="utf-8")` does. `cues::repr` escapes control and
non-printable code points as Python's `repr` does.

## Goldens

`tools/dump_goldens.py` runs against the pinned Python release. Each sentence
line holds the tokens, the proper-noun flags, `is_ambiguous` per token, and the
output of `add_extra_diacritics`. For each homograph token it also holds:

- `guess_pos` and `guess_sense`, with and without the proper-noun bias
- `disambiguate`, the four POS scores and `resolve_sense` for each POS
- `guess_sense` for each tagger hint and `disambiguate` for each POS override
- the scores, prediction and margin of both models, and the model features.

Last, the line holds `add_extra_diacritics` for `pt-BR` and for `pt-UY`, or
null where the result equals the default.

| Corpus | Inputs | Detail kept |
| --- | ---: | --- |
| test_sentences | 3,868 | bifonia `tests/test_sentences.jsonl`, each sentence as given and lowercased, every field |
| hf_test | 60,953 | bifonia `hf/test.jsonl`, `sentence` and `diacritized_sentence`, CRC-32 of the homograph fields |
| frames | 38,430 | every homograph, capitalised, upper-case and each diacritized form, in 42 sentence frames, CRC-32 of the features |
| voc_cues | 68,291 | every term of every `.voc` file in 16 frames around the homographs that read it, CRC-32 of the homograph fields |
| fuzz | 19,322 | seeded random sentences of homographs, cue terms, punctuation, case changes and Unicode, CRC-32 of the homograph fields |
| extras | 372 | the locative contractions before `sede`, left neighbours of 3 code points and more bytes, and tokens that start with `İ`, every field |
| molho_probes | 402 | `tests/fixtures/molho_probes.tsv`, sentences that test the reading of `molho`, every field |
| molho_frames | 10,558 | each term of the 21 word lists that score `molho`, in 30 clause frames, CRC-32 of the features |
| forced | 74,854 | the test sentences, frames, extras and `molho` sentences, and every seventh other sentence, with the model branch forced |

A CRC-32 field is the checksum of the same JSON text. The C++ test computes the
same checksum. The file `calls.jsonl.gz` holds 79,262 direct calls:

- every data table, word list, cue table and model entry
- the infopedia rows and every `ipa` lookup
- each code point that has a case mapping, a decomposition, a combining class, a
  digit value or category Mn, and both ends of each `\w` range
- number probes through `score_adp`, `cue_score` at every index, and every
  index from below `-len` to above `len`
- the lect overrides table, and 127 lect tags through `lect_overrides` and
  through `add_extra_diacritics` on 22 sentences: each lect and sub-lect, case,
  `_`, whitespace, script subtags, and tags that only look like a lect
- a four-letter script subtag at both ends of each `str.isalpha` range
- each public function on hand-built arguments, with each error recorded by its
  Python class
- 3,598 ill-formed UTF-8 byte strings in four contexts, and word lists and cue
  sets with ill-formed members
- `cues::repr` on control, format, private-use and unassigned code points
- the model files in `tests/fixtures/models`: numbers past the double range,
  ties, NaN scores, escapes, and text that Python's `json` rejects
- the `.voc` files in `tests/fixtures/data`: line breaks, comments and ill-formed
  UTF-8.

To regenerate, install the reference release in a fresh virtual environment, then
run:

```
python -I tools/gen_unicode_tables.py > src/unicode_tables.inc
python -I tools/gen_tables.py diacritized > src/diacritized.inc
python -I tools/gen_tables.py default_pos > src/default_pos.inc
python -I tools/dump_goldens.py --out-dir tests/goldens \
    --test-sentences <bifonia>/tests/test_sentences.jsonl \
    --hf-test <bifonia>/hf/test.jsonl
```
