// bifonia: Portuguese heterophonic homograph disambiguation.
//
// A C++17 port of the Python package bifonia, held at byte parity with the
// release named in kPythonReference. Strings are UTF-8. A token list is a
// std::vector<std::string>, and an index into it follows Python's list
// indexing: a negative index counts from the end, and an index out of range
// throws IndexError.
//
// The data files (the homograph table, the word lists and the sense models)
// are read at run time from the data directory. See data_dir().
#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bifonia {

inline constexpr const char* kPythonReference = "bifonia 0.2.0a4";

using Words = std::vector<std::string>;
using WordSet = std::unordered_set<std::string>;

// The Python exception classes that the public functions raise.
struct ValueError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
// Raised when a sense-model file is not valid JSON.
struct JSONDecodeError : ValueError {
    using ValueError::ValueError;
};
struct KeyError : std::out_of_range {
    using std::out_of_range::out_of_range;
};
struct IndexError : std::out_of_range {
    using std::out_of_range::out_of_range;
};
struct FileNotFoundError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A dictionary that keeps insertion order, like a Python dict. Assigning to an
// existing key keeps its position and replaces its value.
template <class V>
class Dict {
public:
    using Item = std::pair<std::string, V>;

    const V* find(const std::string& key) const {
        auto it = index_.find(key);
        return it == index_.end() ? nullptr : &items_[it->second].second;
    }
    V* find(const std::string& key) {
        auto it = index_.find(key);
        return it == index_.end() ? nullptr : &items_[it->second].second;
    }
    bool contains(const std::string& key) const { return index_.count(key) != 0; }
    // Python's d[key]: throws KeyError for a missing key.
    const V& at(const std::string& key) const {
        const V* value = find(key);
        if (value == nullptr) throw KeyError(key);
        return *value;
    }
    // Python's d.setdefault(key, V()).
    V& operator[](const std::string& key) {
        auto it = index_.find(key);
        if (it != index_.end()) return items_[it->second].second;
        index_.emplace(key, items_.size());
        items_.emplace_back(key, V());
        return items_.back().second;
    }
    const std::vector<Item>& items() const { return items_; }
    std::vector<Item>& items() { return items_; }
    size_t size() const { return items_.size(); }
    bool empty() const { return items_.empty(); }

private:
    std::vector<Item> items_;
    std::unordered_map<std::string, size_t> index_;
};

// ── data directory ───────────────────────────────────────────────────────────
// The directory that holds heterophonic_homographs.csv, infopedia_pt.csv, the
// two sense_model_*.json files and locale/<lang>/*.voc. In order of precedence:
// the value given to set_data_dir(), the environment variable
// BIFONIA_DATA_DIR, and the directory compiled in at build time. Each file is
// read once, on first use, and then cached; set the directory before the first
// call.
void set_data_dir(const std::string& path);
std::string data_dir();

// ── bifonia (the package's top level) ───────────────────────────────────────
// Lowercase the text and split it with the pattern -?\w+[.,;:!?]*.
Words tokenize(const std::string& text);
// True when the word, stripped of edge punctuation and lowercased, is a
// known homograph.
bool is_ambiguous(const std::string& word);
// The IPA of the homograph at idx. Throws ValueError when the token is not a
// homograph and KeyError when the sense has no IPA for it.
std::string disambiguate(const Words& words, long idx, const std::optional<std::string>& pos = std::nullopt,
                         const std::optional<std::string>& sense = std::nullopt,
                         const std::optional<std::string>& postag = std::nullopt);
// The meaning slug of the homograph at idx. Empty where Python returns None.
std::optional<std::string> guess_sense(const Words& words, long idx,
                                       const std::optional<std::string>& pos = std::nullopt,
                                       bool proper = false,
                                       const std::optional<std::string>& postag = std::nullopt);
// The lexical class that selects the reading (not a part-of-speech tagger).
std::string guess_pos(const Words& words, long idx, bool proper = false);
// Per token of the pattern above over the original-case text: a likely proper noun.
std::vector<bool> proper_flags(const std::string& text);
// The sentence with the open/closed vowel diacritic on each homograph.
std::string add_extra_diacritics(const std::string& sentence);
// The data tables re-exported at the top level.
const Dict<Dict<std::string>>& HOMOGRAPHS();
const WordSet& AMBIGUOUS_WORDS();
const Dict<Dict<std::vector<std::string>>>& POS_SENSES();
const Dict<Dict<std::string>>& SENSE_POS();

// The diacritic table (Python: the private bifonia._DIACRITIZED) in its dict
// order: each (base, sense) pair and the diacritized form that marks it.
struct DiacriticForm {
    std::string base;
    std::string sense;
    std::string form;
};
const std::vector<DiacriticForm>& DIACRITIZED();

namespace model {
class SenseModel;
}
// Replace the sense model that guess_sense consults (Python: assign
// bifonia._sense_model). A null pointer selects the rules alone.
void set_sense_model(std::shared_ptr<const model::SenseModel> sense_model);

// ── bifonia.version ──────────────────────────────────────────────────────────
namespace version {
inline constexpr int VERSION_MAJOR = 0;
inline constexpr int VERSION_MINOR = 2;
inline constexpr int VERSION_BUILD = 0;
inline constexpr int VERSION_ALPHA = 4;
inline constexpr const char* VERSION_STR = "0.2.0a4";
}  // namespace version

// ── bifonia.vocab ────────────────────────────────────────────────────────────
namespace vocab {
inline constexpr const char* DEFAULT_LANG = "pt-PT";
// The lowercased terms of locale/<lang>/<name>.voc. Throws FileNotFoundError.
const WordSet& voc(const std::string& name, const std::string& lang = DEFAULT_LANG);
}  // namespace vocab

// ── bifonia.data ─────────────────────────────────────────────────────────────
namespace data {
struct Tables {
    Dict<Dict<std::string>> HOMOGRAPHS;                   // word → sense → IPA
    Dict<Dict<std::string>> SENSE_POS;                    // word → sense → first POS
    Dict<Dict<std::vector<std::string>>> SENSE_POSES;     // word → sense → POS list
    Dict<Dict<std::vector<std::string>>> POS_SENSES;      // word → POS → senses
    WordSet AMBIGUOUS_WORDS;
    WordSet POS_AMBIGUOUS;
    Dict<std::string> VERBS_IPA, NOUNS_IPA, ADP_IPA, ADJ_IPA;
    Dict<std::string> DEFAULT_SENSE;
    Dict<std::string> DEFAULT_POS;
    Dict<Dict<int>> BASE_SCORE;
    WordSet STOPPABLE_THINGS, DET, QUANT, PRON, AUX_VERBS, NUMERIC, SOBRE_GOV,
        NEVER_AFTER_PREP, PREP, BEFORE_PREP, AFTER_PREP;
};
// Loaded once from the data directory.
const Tables& tables();
}  // namespace data

// ── bifonia.cues ─────────────────────────────────────────────────────────────
namespace cues {
struct Cue {
    std::string sense;
    std::string voc;
    std::optional<std::string> requires_prev;
};
struct CueRule {
    std::string default_sense;  // Python: CueRule.default
    std::vector<Cue> cues;
};
bool operator==(const Cue& a, const Cue& b);
bool operator==(const CueRule& a, const CueRule& b);
// Python's repr() of the named tuples.
std::string repr(const Cue& cue);
std::string repr(const CueRule& rule);
const Dict<CueRule>& SENSE_CUES();
// voc name → (word, sense)
const Dict<std::pair<std::string, std::string>>& FEATURE_CUES();
}  // namespace cues

// ── bifonia.text ─────────────────────────────────────────────────────────────
namespace text {
// Remove . , ; : ! ? " ' ( ) [ ] { } « » – — anywhere in the token.
std::string strip_punct(const std::string& token);
// NFD, then drop every code point of category Mn.
std::string fold(const std::string& text);
// The accent-folded copy of a cue set.
WordSet folded(const WordSet& cues);
// Proximity-weighted count of folded cues in words (2 within near, else 1).
int cue_score(const Words& words, long idx, const WordSet& cues, long near = 4);
}  // namespace text

// ── bifonia.scoring ──────────────────────────────────────────────────────────
namespace scoring {
int score_adp(const Words& words, long idx);
int score_noun(const Words& words, long idx);
int score_verb(const Words& words, long idx);
int score_adj(const Words& words, long idx);
// Throws ValueError when words[idx] is not a homograph.
std::string guess_pos(const Words& words, long idx, bool proper = false);
std::optional<std::string> resolve_sense(const std::string& word, const Words& words, long idx,
                                         const std::string& pos);
}  // namespace scoring

// ── bifonia.features ─────────────────────────────────────────────────────────
namespace features {
const std::vector<std::string>& STRUCTURAL_VOCS();
// The STRUCTURAL_VOCS sets that exist for lang, in STRUCTURAL_VOCS order.
const Dict<WordSet>& load_structural_vocs(const std::string& lang = "pt-pt");
// The sparse features of the word at idx, in Python's insertion order.
Dict<double> extract_features(const Words& words, long idx, const Dict<WordSet>& vocs);
}  // namespace features

// ── bifonia.model ────────────────────────────────────────────────────────────
namespace model {
std::string NB_PATH();
std::string PERCEPTRON_PATH();

// One sense's weights, sorted by feature name for lookup.
using Weights = std::vector<std::pair<std::string, double>>;

struct WordEntry {
    std::string route = "rules";
    double margin_tau = 0.0;
    std::vector<std::string> senses;
    Dict<double> bias;
    Dict<Weights> weights;
    // The weight of feature for sense, or nullptr.
    const double* weight(const std::string& sense, const std::string& feature) const;
};

class SenseModel {
public:
    // Load and cache the model at path (default: PERCEPTRON_PATH()). Throws
    // FileNotFoundError when the file is missing and JSONDecodeError when it
    // is not valid JSON.
    static std::shared_ptr<const SenseModel> load(const std::optional<std::string>& path = std::nullopt);

    long version = 1;
    std::string model_type = "unknown";
    Dict<std::string> config;  // each value as its JSON text
    std::string lang = "pt-pt";
    Dict<WordEntry> words;

    bool has(const std::string& word) const;
    std::string route(const std::string& word) const;
    double margin_tau(const std::string& word) const;
    // Throws KeyError when the word has no entry.
    Dict<double> scores(const std::string& word, const Words& words, long idx,
                        const Dict<WordSet>* vocs = nullptr) const;
    std::string predict(const std::string& word, const Words& words, long idx,
                        const Dict<WordSet>* vocs = nullptr) const;
    double margin(const std::string& word, const Words& words, long idx,
                  const Dict<WordSet>* vocs = nullptr) const;
};
}  // namespace model

// ── bifonia.infopedia ────────────────────────────────────────────────────────
namespace infopedia {
// One CSV row as (column, value) pairs in header order.
using Row = std::vector<std::pair<std::string, std::string>>;
const Dict<std::vector<Row>>& load();
const std::vector<Row>& readings(const std::string& word);
std::optional<std::string> ipa(const std::string& word, const std::optional<std::string>& pos = std::nullopt);
}  // namespace infopedia

}  // namespace bifonia
