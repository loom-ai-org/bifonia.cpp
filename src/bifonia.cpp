// bifonia/__init__.py: tokenization, the sense and POS guesses, the IPA lookup
// and the diacritic marking of a sentence.
#include <map>
#include <mutex>

#include "detail.hpp"

namespace bifonia {

namespace {
#include "diacritized.inc"

using detail::U32;

constexpr std::string_view kEdgePunct = ".,;:!?";

bool is_edge_punct(char32_t c) { return c < 0x80 && kEdgePunct.find(static_cast<char>(c)) != std::string_view::npos; }

// re.findall(r"-?\w+[.,;:!?]*", text)
std::vector<U32> find_tokens(const U32& text) {
    std::vector<U32> out;
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        size_t j;
        if (text[i] == '-' && i + 1 < n && detail::is_word(text[i + 1])) {
            j = i + 1;
        } else if (detail::is_word(text[i])) {
            j = i;
        } else {
            ++i;
            continue;
        }
        while (j < n && detail::is_word(text[j])) ++j;
        while (j < n && is_edge_punct(text[j])) ++j;
        out.push_back(text.substr(i, j - i));
        i = j;
    }
    return out;
}

// _strip_edge: trailing punctuation and a leading clitic hyphen.
std::string strip_edge(const std::string& token) { return detail::strip_chars(token, ".,;:!?-"); }

struct Diacritics {
    std::map<std::pair<std::string, std::string>, std::string> forms;  // (base, sense) → form
    std::unordered_map<std::string, std::string> to_base;
    std::unordered_map<std::string, std::string> to_sense;
};

const Diacritics& diacritics() {
    static const Diacritics d = [] {
        Diacritics out;
        for (const auto& e : kDiacritized) {
            out.forms[{e.base, e.sense}] = e.form;
            out.to_base[e.form] = e.base;
            out.to_sense[e.form] = e.sense;
        }
        return out;
    }();
    return d;
}

const std::string* lookup(const std::unordered_map<std::string, std::string>& map, const std::string& key) {
    auto it = map.find(key);
    return it == map.end() ? nullptr : &it->second;
}

struct ModelSlot {
    std::mutex mutex;
    bool loaded = false;
    std::shared_ptr<const model::SenseModel> model;
};

ModelSlot& model_slot() {
    static ModelSlot slot;
    return slot;
}

// _sense_model: the perceptron, loaded once; null when it cannot be loaded.
std::shared_ptr<const model::SenseModel> sense_model() {
    ModelSlot& slot = model_slot();
    std::lock_guard<std::mutex> lock(slot.mutex);
    if (!slot.loaded) {
        try {
            slot.model = model::SenseModel::load();
        } catch (const std::exception&) {
            slot.model = nullptr;
        }
        slot.loaded = true;
    }
    return slot.model;
}

std::optional<std::string> rule_sense(const std::string& base, const Words& words, long idx,
                                      const std::optional<std::string>& pos = std::nullopt, bool proper = false) {
    const std::string resolved = (pos && !pos->empty()) ? *pos : scoring::guess_pos(words, idx, proper);
    return scoring::resolve_sense(base, words, idx, resolved);
}

void set_at(Words& words, long idx, const std::string& value) {
    const long n = static_cast<long>(words.size());
    words[static_cast<size_t>(idx < 0 ? idx + n : idx)] = value;
}
}  // namespace

const std::vector<DiacriticForm>& DIACRITIZED() {
    static const std::vector<DiacriticForm> table = [] {
        std::vector<DiacriticForm> out;
        for (const auto& e : kDiacritized) out.push_back({e.base, e.sense, e.form});
        return out;
    }();
    return table;
}

void set_sense_model(std::shared_ptr<const model::SenseModel> sense_model) {
    ModelSlot& slot = model_slot();
    std::lock_guard<std::mutex> lock(slot.mutex);
    slot.model = std::move(sense_model);
    slot.loaded = true;
}

Words tokenize(const std::string& text) {
    Words out;
    for (const auto& t : find_tokens(detail::py_lower(detail::from_utf8(text)))) out.push_back(detail::to_utf8(t));
    return out;
}

bool is_ambiguous(const std::string& word) {
    return detail::contains(data::tables().AMBIGUOUS_WORDS, detail::py_lower(strip_edge(word)));
}

std::string disambiguate(const Words& words, long idx, const std::optional<std::string>& pos,
                         const std::optional<std::string>& sense, const std::optional<std::string>& postag) {
    const std::string& raw = detail::at(words, idx);
    const std::string token = strip_edge(raw);
    const std::string* base = lookup(diacritics().to_base, token);
    const std::string word = base ? *base : token;
    if (!detail::contains(data::tables().AMBIGUOUS_WORDS, word)) {
        throw ValueError("'" + raw + "' is not a known heterophonic homograph");
    }
    const std::optional<std::string> chosen = sense ? sense : guess_sense(words, idx, pos, false, postag);
    if (!chosen) throw KeyError("None");
    return data::tables().HOMOGRAPHS.at(word).at(*chosen);
}

std::optional<std::string> guess_sense(const Words& words, long idx, const std::optional<std::string>& pos,
                                       bool proper, const std::optional<std::string>& postag) {
    const std::string& raw = detail::at(words, idx);
    const std::string token = strip_edge(raw);
    const Diacritics& d = diacritics();
    if (const std::string* sense = lookup(d.to_sense, token)) return *sense;
    const std::string* mapped = lookup(d.to_base, token);
    const std::string base = mapped ? *mapped : token;
    Words copy;
    const Words* normalised = &words;
    if (base != raw) {
        copy = words;
        set_at(copy, idx, base);
        normalised = &copy;
    }
    if (postag) {
        const std::string tag = *postag == "PROPN" ? "NOUN" : *postag == "AUX" ? "VERB" : *postag;
        if (const auto* pmap = data::tables().POS_SENSES.find(base)) {
            const auto* cands = pmap->find(tag);
            if (cands && cands->size() == 1) return (*cands)[0];
        }
    }
    if (pos) return rule_sense(base, *normalised, idx, pos);
    if (proper) return rule_sense(base, *normalised, idx, std::nullopt, true);
    const auto m = sense_model();
    if (m && m->has(base) && m->route(base) == "model" &&
        m->margin(base, *normalised, idx) >= m->margin_tau(base)) {
        return m->predict(base, *normalised, idx);
    }
    return rule_sense(base, *normalised, idx);
}

std::string guess_pos(const Words& words, long idx, bool proper) {
    const std::string& raw = detail::at(words, idx);
    const std::string token = strip_edge(raw);
    const Diacritics& d = diacritics();
    const std::string* mapped = lookup(d.to_base, token);
    const std::string base = mapped ? *mapped : token;
    if (const std::string* sense = lookup(d.to_sense, token)) {
        return data::tables().SENSE_POS.at(base).at(*sense);
    }
    if (base != raw) {
        Words normalised = words;
        set_at(normalised, idx, base);
        return scoring::guess_pos(normalised, idx, proper);
    }
    return scoring::guess_pos(words, idx, proper);
}

std::vector<bool> proper_flags(const std::string& text) {
    const auto toks = find_tokens(detail::from_utf8(text));
    std::vector<bool> flags(toks.size(), false);
    auto lstrip_dash = [](const U32& t) {
        size_t b = 0;
        while (b < t.size() && t[b] == '-') ++b;
        return t.substr(b);
    };
    for (size_t i = 1; i < toks.size(); ++i) {
        const U32 core = lstrip_dash(toks[i]);
        const U32& prev = toks[i - 1];
        const bool boundary =
            !prev.empty() && std::u32string_view(U".!?:…").find(prev.back()) != std::u32string_view::npos;
        if (!core.empty() && detail::is_upper(core[0]) && !boundary) flags[i] = true;
    }
    if (toks.size() > 1) {
        U32 nxt = lstrip_dash(toks[1]);
        size_t b = 0, e = nxt.size();
        while (b < e && is_edge_punct(nxt[b])) ++b;
        while (e > b && is_edge_punct(nxt[e - 1])) --e;
        nxt = nxt.substr(b, e - b);
        const std::string lowered = detail::to_utf8(detail::py_lower(nxt));
        if ((!nxt.empty() && (detail::is_upper(nxt[0]) || detail::is_digit(nxt[0]))) || lowered == "nº" ||
            lowered == "n.º" || lowered == "no" || lowered == "nr") {
            flags[0] = true;
        }
    }
    return flags;
}

std::string add_extra_diacritics(const std::string& sentence) {
    const Words words = tokenize(sentence);
    const std::vector<bool> caps = proper_flags(sentence);
    std::map<size_t, std::string> replacements;
    for (size_t i = 0; i < words.size(); ++i) {
        const std::string base = strip_edge(words[i]);
        if (!is_ambiguous(base)) continue;
        if (i >= caps.size()) throw IndexError("list index out of range");
        const auto sense = guess_sense(words, static_cast<long>(i), std::nullopt, caps[i]);
        if (!sense) continue;
        auto it = diacritics().forms.find({base, *sense});
        if (it != diacritics().forms.end() && !it->second.empty()) replacements[i] = it->second;
    }
    if (replacements.empty()) return detail::to_utf8(detail::from_utf8(sentence));

    // Walk the \w+ runs of the original sentence, keeping each token's case.
    const U32 text = detail::from_utf8(sentence);
    U32 out;
    size_t pos = 0, idx = 0, i = 0;
    while (i < text.size()) {
        if (!detail::is_word(text[i])) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < text.size() && detail::is_word(text[j])) ++j;
        out += text.substr(pos, i - pos);
        auto it = replacements.find(idx);
        if (it == replacements.end()) {
            out += text.substr(i, j - i);
        } else {
            U32 diac = detail::from_utf8(it->second);
            if (detail::is_upper(text[i])) diac = detail::py_upper(diac.substr(0, 1)) + diac.substr(1);
            out += diac;
        }
        pos = j;
        ++idx;
        i = j;
    }
    out += text.substr(pos);
    return detail::to_utf8(out);
}

}  // namespace bifonia
