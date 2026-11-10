// bifonia/text.py and bifonia/cues.py: token normalisation, accent folding,
// cue scoring and the declarative sense-cue registry.
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

#include "detail.hpp"

namespace bifonia::text {

std::string strip_punct(const std::string& token) {
    static const std::u32string kPunct = U".,;:!?\"'()[]{}«»–—";
    bool plain = true;
    for (unsigned char c : token) {
        if (c >= 0x80 || kPunct.find(c) != std::u32string::npos) {
            plain = false;
            break;
        }
    }
    if (plain) return token;
    std::u32string out;
    for (char32_t c : detail::from_utf8(token)) {
        if (kPunct.find(c) == std::u32string::npos) out.push_back(c);
    }
    return detail::to_utf8(out);
}

std::string fold(const std::string& text) {
    bool ascii = true;
    for (unsigned char c : text) ascii = ascii && c < 0x80;
    if (ascii) return text;
    std::u32string out;
    for (char32_t c : detail::nfd(detail::from_utf8(text))) {
        if (!detail::is_mn(c)) out.push_back(c);
    }
    return detail::to_utf8(out);
}

WordSet folded(const WordSet& cues) {
    WordSet out;
    for (const auto& c : cues) out.insert(fold(c));
    return out;
}

int cue_score(const Words& words, long idx, const WordSet& cues, long near) {
    int score = 0;
    for (size_t j = 0; j < words.size(); ++j) {
        const long jl = static_cast<long>(j);
        if (jl != idx && detail::contains(cues, fold(strip_punct(words[j])))) {
            score += std::labs(jl - idx) <= near ? 2 : 1;
        }
    }
    return score;
}

}  // namespace bifonia::text

namespace bifonia::detail {
// Python's repr() of a str: single quotes unless the text holds a single quote
// and no double quote; control and non-printable code points as escapes.
std::string str_repr(const std::string& s) {
    const char q = (s.find('\'') != std::string::npos && s.find('"') == std::string::npos) ? '"' : '\'';
    std::string out(1, q);
    char buf[16];
    for (char32_t c : from_utf8(s)) {
        if (c == '\\' || c == static_cast<char32_t>(q)) {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c < 0x20 || c == 0x7F) {
            std::snprintf(buf, sizeof buf, "\\x%02x", static_cast<unsigned>(c));
            out += buf;
        } else if (c < 0x7F || is_printable(c)) {
            out += to_utf8(std::u32string(1, c));
        } else {
            const char* format = c <= 0xFF ? "\\x%02x" : c <= 0xFFFF ? "\\u%04x" : "\\U%08x";
            std::snprintf(buf, sizeof buf, format, static_cast<unsigned>(c));
            out += buf;
        }
    }
    return out + q;
}

// The folded copy of a cached voc, built once (Python's lru_cache on folded).
const WordSet& folded_voc(const std::string& name) {
    static std::mutex mutex;
    static std::map<std::string, WordSet> cache;
    const WordSet* source = &vocab::voc(name);
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(name);
    if (it == cache.end()) it = cache.emplace(name, text::folded(*source)).first;
    return it->second;
}
}  // namespace bifonia::detail

namespace bifonia::cues {

const Dict<CueRule>& SENSE_CUES() {
    static const Dict<CueRule> rules = [] {
        Dict<CueRule> r;
        r["bola"] = CueRule{"ball", {Cue{"loaf", "bola_loaf_cues", {}}, Cue{"ball", "bola_ball_cues", {}}}};
        r["lobo"] = CueRule{"wolf", {Cue{"lobe", "lobo_lobe_cues", {}}}};
        r["polo"] = CueRule{"pole", {Cue{"fledgling", "polo_fledgling_cues", {}}}};
        r["cor"] = CueRule{"colour", {Cue{"by_heart", "cor_memory_cues", std::string("de")}}};
        r["forma"] = CueRule{"shape", {Cue{"mould", "mould_cues", {}}}};
        return r;
    }();
    return rules;
}

const Dict<std::pair<std::string, std::string>>& FEATURE_CUES() {
    static const Dict<std::pair<std::string, std::string>> cues = [] {
        Dict<std::pair<std::string, std::string>> c;
        c["bola_loaf_cues"] = {"bola", "loaf"};
        c["bola_ball_cues"] = {"bola", "ball"};
        c["lobo_lobe_cues"] = {"lobo", "lobe"};
        c["polo_fledgling_cues"] = {"polo", "fledgling"};
        c["cor_memory_cues"] = {"cor", "by_heart"};
        c["mould_cues"] = {"forma", "mould"};
        c["sede_seat_cues"] = {"sede", "seat"};
        c["sede_thirst_cues"] = {"sede", "thirst"};
        c["bundle_things"] = {"molho", "bundle"};
        c["soak_verbs"] = {"molho", "soak"};
        c["cut_context"] = {"corte", "cut"};
        return c;
    }();
    return cues;
}


bool operator==(const Cue& a, const Cue& b) {
    return a.sense == b.sense && a.voc == b.voc && a.requires_prev == b.requires_prev;
}

bool operator==(const CueRule& a, const CueRule& b) {
    return a.default_sense == b.default_sense && a.cues == b.cues;
}

std::string repr(const Cue& cue) {
    return "Cue(sense=" + detail::str_repr(cue.sense) + ", voc=" + detail::str_repr(cue.voc) +
           ", requires_prev=" + (cue.requires_prev ? detail::str_repr(*cue.requires_prev) : "None") + ")";
}

std::string repr(const CueRule& rule) {
    std::string cues = "(";
    for (size_t i = 0; i < rule.cues.size(); ++i) cues += (i ? ", " : "") + repr(rule.cues[i]);
    if (rule.cues.size() == 1) cues += ",";
    return "CueRule(default=" + detail::str_repr(rule.default_sense) + ", cues=" + cues + "))";
}

}  // namespace bifonia::cues
