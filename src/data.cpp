// bifonia/data.py: the homograph table, its POS indexes and the context word lists.
#include <algorithm>

#include "detail.hpp"

namespace bifonia::data {

namespace {
#include "default_pos.inc"

std::vector<std::string> split_pos(const std::string& field) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t bar = field.find('|', start);
        std::string p = field.substr(start, bar == std::string::npos ? std::string::npos : bar - start);
        if (!p.empty()) out.push_back(p);
        if (bar == std::string::npos) break;
        start = bar + 1;
    }
    return out;
}

const std::string& column(const std::vector<std::pair<std::string, std::string>>& row, const char* name) {
    for (const auto& [key, value] : row) {
        if (key == name) return value;
    }
    throw KeyError(name);
}

WordSet set_of(const char* name) { return vocab::voc(name); }

// _ipa_by_pos: word → IPA of the first sense that carries pos.
Dict<std::string> ipa_by_pos(const Tables& t, const std::string& pos) {
    Dict<std::string> out;
    for (const auto& [word, pmap] : t.POS_SENSES.items()) {
        for (const auto& [p, senses] : pmap.items()) {
            if (p == pos) out[word] = t.HOMOGRAPHS.at(word).at(senses[0]);
        }
    }
    return out;
}

Tables build() {
    Tables t;
    std::string text;
    const std::string path = detail::data_path("heterophonic_homographs.csv");
    if (!detail::read_file(path, text)) throw FileNotFoundError("cannot read " + path);
    for (const auto& row : detail::read_csv(text)) {
        const std::string& w = column(row, "word");
        const std::string& s = column(row, "sense");
        const std::string& ipa = column(row, "ipa");
        const auto poses = split_pos(column(row, "pos"));
        if (poses.empty()) throw IndexError("tuple index out of range");
        t.HOMOGRAPHS[w][s] = ipa;
        t.SENSE_POS[w][s] = poses[0];
        t.SENSE_POSES[w][s] = poses;
        for (const auto& p : poses) t.POS_SENSES[w][p].push_back(s);
    }
    for (const auto& item : t.HOMOGRAPHS.items()) t.AMBIGUOUS_WORDS.insert(item.first);
    for (const auto& [word, pmap] : t.POS_SENSES.items()) {
        for (const auto& item : pmap.items()) {
            if (item.second.size() > 1) t.POS_AMBIGUOUS.insert(word);
        }
    }
    t.VERBS_IPA = ipa_by_pos(t, "VERB");
    t.NOUNS_IPA = ipa_by_pos(t, "NOUN");
    t.ADP_IPA = ipa_by_pos(t, "ADP");
    t.ADJ_IPA = ipa_by_pos(t, "ADJ");

    for (const auto& [word, sense] : {std::pair<const char*, const char*>{"sede", "thirst"},
                                      {"bola", "ball"}, {"cor", "colour"}, {"lobo", "wolf"},
                                      {"polo", "pole"}, {"corte", "cut"}, {"forma", "shape"},
                                      {"molho", "sauce"}, {"tola", "foolish"}}) {
        t.DEFAULT_SENSE[word] = sense;
    }
    for (const auto& e : kDefaultPos) t.DEFAULT_POS[e.word] = e.pos;
    bool same = t.DEFAULT_POS.size() == t.AMBIGUOUS_WORDS.size();
    for (const auto& item : t.DEFAULT_POS.items()) same = same && detail::contains(t.AMBIGUOUS_WORDS, item.first);
    if (!same) throw std::logic_error("DEFAULT_POS keys differ from AMBIGUOUS_WORDS");

    auto& pelo = t.BASE_SCORE["pelo"];
    pelo["ADP"] = 1;
    pelo["NOUN"] = 0;
    pelo["VERB"] = 0;

    t.STOPPABLE_THINGS = set_of("stoppable_things");
    t.DET = set_of("determiners");
    t.QUANT = set_of("quantifiers");
    t.PRON = set_of("pronouns");
    t.AUX_VERBS = set_of("aux_verbs");
    t.NUMERIC = set_of("numeric");
    t.SOBRE_GOV = set_of("sobre_governors");
    t.NEVER_AFTER_PREP = set_of("never_after_prep");
    t.PREP = set_of("ambiguous_preps");
    t.BEFORE_PREP = set_of("before_prep_extra");
    t.BEFORE_PREP.insert(t.AUX_VERBS.begin(), t.AUX_VERBS.end());
    t.AFTER_PREP = set_of("after_prep_extra");
    t.AFTER_PREP.insert(t.DET.begin(), t.DET.end());
    t.AFTER_PREP.insert(t.PRON.begin(), t.PRON.end());
    for (const auto& w : set_of("after_prep_exclude")) t.AFTER_PREP.erase(w);

    const std::string lect_path = detail::data_path("lect_overrides.csv");
    if (!detail::read_file(lect_path, text)) throw FileNotFoundError("cannot read " + lect_path);
    for (const auto& row : detail::read_csv(text)) {
        const std::string& lang = column(row, "lang");
        const std::string& word = column(row, "word");
        t.LECT_OVERRIDES[lang][word] = column(row, "form");
    }
    return t;
}
}  // namespace

const Tables& tables() {
    static const Tables t = build();
    return t;
}

const Dict<std::string>& lect_overrides(const std::optional<std::string>& lang) {
    static const Dict<std::string> kNone;
    if (!lang || lang->empty()) return kNone;
    detail::U32 text = detail::from_utf8(detail::py_strip(*lang));
    std::replace(text.begin(), text.end(), U'_', U'-');
    text = detail::py_lower(text);
    std::vector<detail::U32> subtags{{}};
    for (char32_t c : text) {
        if (c == U'-') {
            subtags.emplace_back();
        } else {
            subtags.back().push_back(c);
        }
    }
    // A BCP 47 script subtag, such as "latn", does not change the lect.
    if (subtags.size() > 1 && subtags[1].size() == 4 &&
        std::all_of(subtags[1].begin(), subtags[1].end(), detail::is_alpha)) {
        subtags.erase(subtags.begin() + 1);
    }
    detail::U32 tag;
    for (size_t i = 0; i < subtags.size(); ++i) tag += (i ? U"-" : U"") + subtags[i];
    for (const auto& [lect, overrides] : tables().LECT_OVERRIDES.items()) {
        const detail::U32 name = detail::py_lower(detail::from_utf8(lect));
        if (tag == name || tag.compare(0, name.size() + 1, name + U"-") == 0) return overrides;
    }
    return kNone;
}

}  // namespace bifonia::data

namespace bifonia {
const Dict<Dict<std::string>>& HOMOGRAPHS() { return data::tables().HOMOGRAPHS; }
const WordSet& AMBIGUOUS_WORDS() { return data::tables().AMBIGUOUS_WORDS; }
const Dict<Dict<std::vector<std::string>>>& POS_SENSES() { return data::tables().POS_SENSES; }
const Dict<Dict<std::string>>& SENSE_POS() { return data::tables().SENSE_POS; }
const Dict<std::string>& lect_overrides(const std::optional<std::string>& lang) { return data::lect_overrides(lang); }
}  // namespace bifonia
