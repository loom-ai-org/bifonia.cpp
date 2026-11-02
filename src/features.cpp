// bifonia/features.py: the sparse feature extractor of the sense models.
#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>

#include "detail.hpp"

namespace bifonia::features {

namespace {
// Stripped, lowercased token at i, or "" when i is out of range.
std::string tok(const Words& words, long i) {
    return (i >= 0 && i < static_cast<long>(words.size()))
               ? detail::py_lower(text::strip_punct(words[static_cast<size_t>(i)]))
               : "";
}

bool ends_with_any(const std::string& w, std::initializer_list<const char*> suffixes) {
    for (const char* s : suffixes) {
        if (detail::ends_with(w, s)) return true;
    }
    return false;
}

bool is_inf(const std::string& w) { return ends_with_any(w, {"ar", "er", "ir"}) && detail::cp_len(w) > 2; }

bool looks_verb(const std::string& w) {
    if (ends_with_any(w, {"ou", "eu", "iu", "ei", "ava", "ara", "era"})) return true;
    if (detail::ends_with(w, "ia")) {
        const auto u = detail::from_utf8(w);
        if (u.size() >= 4 && std::u32string_view(U"aeiouáéíóúâêîôûãõàèìòùäëïöü").find(u[u.size() - 3]) ==
                                 std::u32string_view::npos) {
            return true;
        }
    }
    return false;
}
}  // namespace

const std::vector<std::string>& STRUCTURAL_VOCS() {
    static const std::vector<std::string> names = {
        "determiners", "pronouns", "quantifiers", "articles", "sing_articles",
        "contracted_det", "de_contractions", "locative_contractions", "contracted_a",
        "copula", "aux_verbs", "passive_aux", "post_clitics", "clitics_all",
        "comparatives", "intensifiers", "neg_adv", "pure_negation",
        "temporal_conj", "conj_subjunctive", "numeric", "adv_bridge",
        "after_prep_extra", "after_prep_weak", "never_after_prep", "before_prep_extra",
        "exclamative_det", "function_words", "fem_det", "masc_det",
        "prep_governing", "optative_adv", "material_prep",
    };
    return names;
}

const Dict<WordSet>& load_structural_vocs(const std::string& lang) {
    static std::mutex mutex;
    static std::map<std::string, std::unique_ptr<Dict<WordSet>>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto& slot = cache[lang];
    if (slot) return *slot;
    // Built aside, so that a file that fails to decode leaves nothing cached.
    auto vocs = std::make_unique<Dict<WordSet>>();
    const std::filesystem::path dir = std::filesystem::path(data_dir()) / "locale" / detail::py_lower(lang);
    for (const auto& name : STRUCTURAL_VOCS()) {
        std::string text;
        std::error_code ec;
        const auto path = dir / (name + ".voc");
        if (std::filesystem::exists(path, ec) && detail::read_file(path.string(), text)) {
            (*vocs)[name] = detail::parse_voc(text);
        }
    }
    slot = std::move(vocs);
    return *slot;
}

Dict<double> extract_features(const Words& words, long idx, const Dict<WordSet>& vocs) {
    Dict<double> feats;
    const long n = static_cast<long>(words.size());
    const std::string self_tok = tok(words, idx);

    for (long d = 1; d <= 4; ++d) {
        const long i = idx - d;
        feats["L" + std::to_string(d) + "=" + (i >= 0 ? tok(words, i) : "<BOS>")] = 1.0;
    }
    for (long d = 1; d <= 3; ++d) {
        const long i = idx + d;
        feats["R" + std::to_string(d) + "=" + (i < n ? tok(words, i) : "<EOS>")] = 1.0;
    }

    for (long i = std::max(0L, idx - 4); i < std::min(n, idx + 5); ++i) {
        if (i == idx) continue;
        const std::string t = tok(words, i);
        if (!t.empty()) feats["W=" + t] += 1.0;
    }

    const std::pair<const char*, long> slots[] = {
        {"L1", idx - 1}, {"L2", idx - 2}, {"R1", idx + 1}, {"R2", idx + 2}, {"S", idx}};
    for (const auto& [slot, i] : slots) {
        const std::string t = tok(words, i);
        if (t.empty()) continue;
        for (const auto& [name, members] : vocs.items()) {
            if (detail::contains(members, t)) feats[std::string(slot) + "∈" + name] = 1.0;
        }
    }

    if (is_inf(self_tok)) feats["self_sfx=INF"] = 1.0;
    const auto self_u = detail::from_utf8(self_tok);
    if (self_u.size() >= 3) {
        feats["self_sfx3=" + detail::to_utf8(std::u32string_view(self_u).substr(self_u.size() - 3))] = 1.0;
        feats["self_pfx3=" + detail::to_utf8(std::u32string_view(self_u).substr(0, 3))] = 1.0;
    }
    const std::string r1 = tok(words, idx + 1);
    const std::string l1 = tok(words, idx - 1);
    if (detail::ends_with(r1, "mente")) feats["R1_mente"] = 1.0;
    if (ends_with_any(r1, {"ção", "são", "gem", "ura", "ência", "ância", "mento", "ismo", "ise"})) {
        feats["R1_deverbal"] = 1.0;
    }
    if (is_inf(l1)) feats["L1_inf"] = 1.0;
    if (is_inf(r1)) feats["R1_inf"] = 1.0;
    if (idx == 0) feats["pos0"] = 1.0;
    if (!l1.empty() && looks_verb(l1)) feats["L1_looks_verb"] = 1.0;

    for (const auto& [voc_name, cue] : cues::FEATURE_CUES().items()) {
        if (cue.first == self_tok) {
            const int score = text::cue_score(words, idx, detail::folded_voc(voc_name));
            if (score) feats["CUE:" + cue.second] = static_cast<double>(score);
        }
    }
    return feats;
}

}  // namespace bifonia::features
