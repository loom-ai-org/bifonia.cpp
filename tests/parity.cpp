// Byte parity with the Python bifonia. Each golden line holds an input and
// Python's output for the public functions; this recomputes the line in C++
// and compares the bytes.
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "bifonia/bifonia.hpp"
#include "json_line.hpp"

namespace bf = bifonia;
using json_line::quote;
using json_line::Value;

namespace {

// ── JSON pieces, written as Python's json.dumps writes them ──────────────────

std::string J(const std::string& s) { return quote(s); }
std::string J(const char* s) { return quote(s); }
std::string J(const std::optional<std::string>& s) { return s ? quote(*s) : "null"; }
std::string J(bool b) { return b ? "true" : "false"; }
std::string J(int v) { return std::to_string(v); }
std::string J(long v) { return std::to_string(v); }
std::string J(unsigned long v) { return std::to_string(v); }
std::string J(const std::vector<std::string>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? "," : "") + quote(items[i]);
    return out + "]";
}

std::string list(const std::vector<std::string>& parts) {
    std::string out = "[";
    for (size_t i = 0; i < parts.size(); ++i) out += (i ? "," : "") + parts[i];
    return out + "]";
}

std::vector<std::string> sorted(const bf::WordSet& set) {
    std::vector<std::string> out(set.begin(), set.end());
    std::sort(out.begin(), out.end());
    return out;
}

std::string bits(const std::vector<bool>& flags) {
    std::string out;
    for (bool f : flags) out.push_back(f ? '1' : '0');
    return out;
}

// Python's repr(float): the shortest round-trip digits, in fixed notation for
// a decimal exponent in [-4, 16) and in exponent notation otherwise.
std::string repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
    std::string s(buf, res.ptr);
    std::string sign;
    if (s[0] == '-') {
        sign = "-";
        s.erase(0, 1);
    }
    const size_t e = s.find('e');
    const int exp = std::stoi(s.substr(e + 1));
    std::string digits;
    for (char c : s.substr(0, e)) {
        if (c != '.') digits.push_back(c);
    }
    if (exp >= -4 && exp < 16) {
        if (exp < 0) return sign + "0." + std::string(static_cast<size_t>(-exp - 1), '0') + digits;
        const size_t int_len = static_cast<size_t>(exp) + 1;
        if (digits.size() <= int_len) return sign + digits + std::string(int_len - digits.size(), '0') + ".0";
        return sign + digits.substr(0, int_len) + "." + digits.substr(int_len);
    }
    std::string mantissa = digits.substr(0, 1);
    if (digits.size() > 1) mantissa += "." + digits.substr(1);
    const int a = std::abs(exp);
    return sign + mantissa + "e" + (exp < 0 ? "-" : "+") + (a < 10 ? "0" : "") + std::to_string(a);
}

// Run f; an exception becomes {"err":"<Python class>"}.
std::string guard(const std::function<std::string()>& f) {
    try {
        return f();
    } catch (const bf::IndexError&) {
        return "{\"err\":\"IndexError\"}";
    } catch (const bf::KeyError&) {
        return "{\"err\":\"KeyError\"}";
    } catch (const bf::JSONDecodeError&) {
        return "{\"err\":\"JSONDecodeError\"}";
    } catch (const bf::UnicodeDecodeError&) {
        return "{\"err\":\"UnicodeDecodeError\"}";
    } catch (const bf::ValueError&) {
        return "{\"err\":\"ValueError\"}";
    } catch (const bf::FileNotFoundError&) {
        return "{\"err\":\"FileNotFoundError\"}";
    } catch (const std::exception& e) {
        return std::string("{\"err\":\"C++ ") + e.what() + "\"}";
    }
}

const std::map<std::string, std::string>& to_base() {
    static const std::map<std::string, std::string> map = [] {
        std::map<std::string, std::string> m;
        for (const auto& d : bf::DIACRITIZED()) m[d.form] = d.base;
        return m;
    }();
    return map;
}

std::string strip_edge(const std::string& token) {
    const std::string chars = ".,;:!?-";
    size_t b = 0, e = token.size();
    while (b < e && chars.find(token[b]) != std::string::npos) ++b;
    while (e > b && chars.find(token[e - 1]) != std::string::npos) --e;
    return token.substr(b, e - b);
}

std::vector<std::pair<long, std::string>> candidates(const bf::Words& words) {
    std::vector<std::pair<long, std::string>> out;
    for (size_t i = 0; i < words.size(); ++i) {
        const std::string token = strip_edge(words[i]);
        if (bf::AMBIGUOUS_WORDS().count(token) || to_base().count(token)) {
            out.emplace_back(static_cast<long>(i), token);
        }
    }
    return out;
}

std::string feature_list(const bf::Words& words, long idx) {
    std::vector<std::string> parts;
    const auto feats = bf::features::extract_features(words, idx, bf::features::load_structural_vocs());
    for (const auto& [k, v] : feats.items()) {
        parts.push_back("[" + J(k) + "," + J(repr(v)) + "]");
    }
    return list(parts);
}

std::string model_record(const bf::model::SenseModel& m, const std::string& base, const bf::Words& words, long idx) {
    if (!m.has(base)) return "null";
    std::vector<std::string> scores;
    const auto values = m.scores(base, words, idx);
    for (const auto& [s, v] : values.items()) scores.push_back("[" + J(s) + "," + J(repr(v)) + "]");
    return list({J(m.route(base)), J(repr(m.margin_tau(base))), list(scores), J(m.predict(base, words, idx)),
                 J(repr(m.margin(base, words, idx)))});
}

const char* const kPoses[] = {"ADP", "NOUN", "VERB", "ADJ"};
const char* const kPostags[] = {"ADP", "NOUN", "VERB", "ADJ", "PROPN", "AUX", "X"};

std::string sentence_record(const std::string& s, int detail) {
    const auto nb = bf::model::SenseModel::load(bf::model::NB_PATH());
    const auto perceptron = bf::model::SenseModel::load(bf::model::PERCEPTRON_PATH());
    const bf::Words words = bf::tokenize(s);
    std::string ambiguous;
    for (const auto& w : words) ambiguous.push_back(bf::is_ambiguous(w) ? '1' : '0');
    std::string out = "[" + J(s) + "," + J(words) + "," + J(bits(bf::proper_flags(s))) + "," + J(ambiguous) + "," +
                      guard([&] { return J(bf::add_extra_diacritics(s)); }) + ",";
    std::vector<std::string> cands;
    for (const auto& item : candidates(words)) {
        const long i = item.first;
        const std::string& token = item.second;
        auto it = to_base().find(token);
        const std::string base = it == to_base().end() ? token : it->second;
        bf::Words norm = words;
        norm[static_cast<size_t>(i)] = base;
        const std::string feats = feature_list(norm, i);
        std::vector<std::string> scores, resolved, tagged, by_pos;
        for (auto f : {bf::scoring::score_adp, bf::scoring::score_noun, bf::scoring::score_verb, bf::scoring::score_adj}) {
            scores.push_back(guard([&] { return J(f(norm, i)); }));
        }
        for (const char* p : kPoses) {
            resolved.push_back(guard([&] { return J(bf::scoring::resolve_sense(base, norm, i, p)); }));
            by_pos.push_back(guard([&] { return J(bf::disambiguate(words, i, std::string(p))); }));
        }
        for (const char* t : kPostags) {
            tagged.push_back(guard([&] { return J(bf::guess_sense(words, i, std::nullopt, false, std::string(t))); }));
        }
        cands.push_back(list({
            J(i),
            guard([&] { return J(bf::guess_pos(words, i)); }),
            guard([&] { return J(bf::guess_pos(words, i, true)); }),
            guard([&] { return J(bf::guess_sense(words, i)); }),
            guard([&] { return J(bf::guess_sense(words, i, std::nullopt, true)); }),
            guard([&] { return J(bf::disambiguate(words, i)); }),
            list(scores),
            list(resolved),
            list(tagged),
            list(by_pos),
            model_record(*nb, base, norm, i),
            model_record(*perceptron, base, norm, i),
            detail == 2 ? feats : J(json_line::crc32_of(feats)),
        }));
    }
    const std::string all = list(cands);
    out += (detail ? all : J(json_line::crc32_of(all))) + ",";
    // The sentence for each lect with overrides, or null where it equals the default.
    const std::string default_text = guard([&] { return J(bf::add_extra_diacritics(s)); });
    std::vector<std::string> lects;
    for (const char* lang : {"pt-BR", "pt-UY"}) {
        const std::string got = guard([&] { return J(bf::add_extra_diacritics(s, std::string(lang))); });
        lects.push_back(got == default_text ? "null" : got);
    }
    return out + list(lects) + "]";
}

std::string forced_record(const std::string& s) {
    const bf::Words words = bf::tokenize(s);
    std::vector<std::string> cands;
    for (const auto& item : candidates(words)) {
        const long i = item.first;
        cands.push_back(list({J(i), guard([&] { return J(bf::guess_sense(words, i)); }),
                              guard([&] { return J(bf::disambiguate(words, i)); })}));
    }
    return "[" + J(s) + "," + guard([&] { return J(bf::add_extra_diacritics(s)); }) + "," + list(cands) + "]";
}

// ── direct calls ─────────────────────────────────────────────────────────────

template <class V, class F>
std::string dict(const bf::Dict<V>& d, F value) {
    std::string out = "{";
    bool first = true;
    for (const auto& [k, v] : d.items()) {
        out += (first ? "" : ",") + J(k) + ":" + value(v);
        first = false;
    }
    return out + "}";
}

template <class V, class F>
std::string items(const bf::Dict<V>& d, F value) {
    std::vector<std::string> parts;
    for (const auto& [k, v] : d.items()) parts.push_back("[" + J(k) + "," + value(v) + "]");
    return list(parts);
}

const bf::WordSet& data_set(const std::string& name) {
    const auto& t = bf::data::tables();
    static const std::map<std::string, const bf::WordSet*> sets = {
        {"AMBIGUOUS_WORDS", &t.AMBIGUOUS_WORDS}, {"POS_AMBIGUOUS", &t.POS_AMBIGUOUS},
        {"STOPPABLE_THINGS", &t.STOPPABLE_THINGS}, {"DET", &t.DET}, {"QUANT", &t.QUANT}, {"PRON", &t.PRON},
        {"AUX_VERBS", &t.AUX_VERBS}, {"NUMERIC", &t.NUMERIC}, {"SOBRE_GOV", &t.SOBRE_GOV},
        {"NEVER_AFTER_PREP", &t.NEVER_AFTER_PREP}, {"PREP", &t.PREP}, {"BEFORE_PREP", &t.BEFORE_PREP},
        {"AFTER_PREP", &t.AFTER_PREP}};
    return *sets.at(name);
}

const bf::Dict<std::string>& data_dict(const std::string& name) {
    const auto& t = bf::data::tables();
    static const std::map<std::string, const bf::Dict<std::string>*> dicts = {
        {"VERBS_IPA", &t.VERBS_IPA}, {"NOUNS_IPA", &t.NOUNS_IPA}, {"ADP_IPA", &t.ADP_IPA},
        {"ADJ_IPA", &t.ADJ_IPA}, {"DEFAULT_SENSE", &t.DEFAULT_SENSE}, {"DEFAULT_POS", &t.DEFAULT_POS}};
    return *dicts.at(name);
}

std::string four_scores(const bf::Words& words, long idx) {
    std::vector<std::string> parts;
    for (auto f : {bf::scoring::score_adp, bf::scoring::score_noun, bf::scoring::score_verb, bf::scoring::score_adj}) {
        parts.push_back(guard([&] { return J(f(words, idx)); }));
    }
    return list(parts);
}

std::shared_ptr<const bf::model::SenseModel> model_named(const std::string& name) {
    return bf::model::SenseModel::load(name == "nb" ? bf::model::NB_PATH() : bf::model::PERCEPTRON_PATH());
}

std::string bools(const std::vector<bool>& flags) {
    std::vector<std::string> parts;
    for (bool f : flags) parts.push_back(J(f));
    return list(parts);
}

std::string reprs(const bf::Dict<double>& d) {
    std::vector<std::string> parts;
    for (const auto& [k, v] : d.items()) parts.push_back("[" + J(k) + "," + J(repr(v)) + "]");
    return list(parts);
}

std::string rows(const std::vector<bf::infopedia::Row>& rs) {
    std::vector<std::string> parts;
    for (const auto& row : rs) {
        std::vector<std::string> cells;
        for (const auto& [k, x] : row) cells.push_back(J(std::vector<std::string>{k, x}));
        parts.push_back(list(cells));
    }
    return list(parts);
}

std::optional<std::string> opt(const Value& v) {
    if (v.is_null()) return std::nullopt;
    return v.str;
}

bf::WordSet set_of(const Value& v) {
    bf::WordSet out;
    for (const auto& item : v.items) out.insert(item.str);
    return out;
}

bf::Dict<bf::WordSet> vocs_of(const Value& v) {
    bf::Dict<bf::WordSet> out;
    for (const auto& pair : v.items) out[pair[0].str] = set_of(pair[1]);
    return out;
}

std::string probe_result(const std::string& fn, const Value& a) {
    if (fn == "tokenize") return J(bf::tokenize(a[0].str));
    if (fn == "proper_flags") return bools(bf::proper_flags(a[0].str));
    if (fn == "add_extra_diacritics") return J(bf::add_extra_diacritics(a[0].str));
    if (fn == "is_ambiguous") return J(bf::is_ambiguous(a[0].str));
    if (fn == "strip_punct") return J(bf::text::strip_punct(a[0].str));
    if (fn == "fold") return J(bf::text::fold(a[0].str));
    if (fn == "infopedia.readings") return rows(bf::infopedia::readings(a[0].str));
    if (fn == "infopedia.ipa") return J(bf::infopedia::ipa(a[0].str, opt(a[1])));
    if (fn == "voc") return J(sorted(bf::vocab::voc(a[0].str, a[1].str)));
    if (fn == "load_structural_vocs") {
        return items(bf::features::load_structural_vocs(a[0].str), [](const auto& x) { return J(sorted(x)); });
    }
    if (fn == "folded") return J(sorted(bf::text::folded(set_of(a[0]))));
    if (fn == "model.load") {
        return J(bf::model::SenseModel::load(bf::data_dir() + "/" + a[0].str)->model_type);
    }
    if (fn == "cue_repr") {
        const auto& rule = bf::cues::SENSE_CUES().at(a[0].str);
        std::vector<std::string> out = {bf::cues::repr(rule)};
        for (const auto& c : rule.cues) out.push_back(bf::cues::repr(c));
        return J(out);
    }
    if (fn == "model.has" || fn == "model.route" || fn == "model.margin_tau") {
        const auto m = model_named(a[0].str);
        if (fn == "model.has") return J(m->has(a[1].str));
        if (fn == "model.route") return J(m->route(a[1].str));
        return J(repr(m->margin_tau(a[1].str)));
    }
    if (fn == "resolve_sense") {
        return J(bf::scoring::resolve_sense(a[0].str, a[1].strings(), static_cast<long>(a[2].integer), a[3].str));
    }
    const bf::Words words = a[fn.rfind("model.", 0) == 0 ? 2 : 0].strings();
    const long idx = static_cast<long>(a[1].integer);
    if (fn == "cue_score") {
        return J(bf::text::cue_score(words, idx, set_of(a[2]), static_cast<long>(a[3].integer)));
    }
    if (fn == "score_adp") return J(bf::scoring::score_adp(words, idx));
    if (fn == "score_noun") return J(bf::scoring::score_noun(words, idx));
    if (fn == "score_verb") return J(bf::scoring::score_verb(words, idx));
    if (fn == "score_adj") return J(bf::scoring::score_adj(words, idx));
    if (fn == "scoring.guess_pos") return J(bf::scoring::guess_pos(words, idx, a[2].boolean));
    if (fn == "guess_pos") return J(bf::guess_pos(words, idx, a[2].boolean));
    if (fn == "guess_sense") return J(bf::guess_sense(words, idx, opt(a[2]), a[3].boolean, opt(a[4])));
    if (fn == "disambiguate") return J(bf::disambiguate(words, idx, opt(a[2]), opt(a[3]), opt(a[4])));
    if (fn == "extract_features") return reprs(bf::features::extract_features(words, idx, vocs_of(a[2])));
    if (fn.rfind("model.", 0) == 0) {
        const auto m = model_named(a[0].str);
        const std::string word = a[1].str;
        const long i = static_cast<long>(a[3].integer);
        const bf::Dict<bf::WordSet> custom = a[4].is_null() ? bf::Dict<bf::WordSet>() : vocs_of(a[4]);
        const bf::Dict<bf::WordSet>* vocs = a[4].is_null() ? nullptr : &custom;
        if (fn == "model.scores") return reprs(m->scores(word, words, i, vocs));
        if (fn == "model.predict") return J(m->predict(word, words, i, vocs));
        if (fn == "model.margin") return J(repr(m->margin(word, words, i, vocs)));
    }
    return "unknown probe " + fn;
}

std::string from_hex(const std::string& hex) {
    std::string out;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) out.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    return out;
}

std::string fixtures_dir;

// Python's repr() of the value a JSON text holds: an integer, a float, or a
// plain ASCII string (the fixtures hold no other kinds).
std::string json_repr(const std::string& raw) {
    if (raw[0] == '"') return "'" + json_line::parse(raw).str + "'";
    if (raw == "NaN") return "nan";
    if (raw == "Infinity") return "inf";
    if (raw == "-Infinity") return "-inf";
    if (raw.find_first_of(".eE") == std::string::npos) {
        const bool zero = raw.find_first_not_of("-0") == std::string::npos;
        return zero ? "0" : raw;
    }
    return repr(std::strtod(raw.c_str(), nullptr));
}

std::string fixture_runs(const bf::model::SenseModel& m, const std::string& word, const Value& runs) {
    std::vector<std::string> out;
    for (const auto& run : runs.items) {
        const std::string fn = run[0].str;
        const bf::Words words = run[1].strings();
        const long idx = static_cast<long>(run[2].integer);
        const std::string got = guard([&] {
            if (fn == "scores") return reprs(m.scores(word, words, idx));
            if (fn == "predict") return J(m.predict(word, words, idx));
            return J(repr(m.margin(word, words, idx)));
        });
        out.push_back(list({J(fn), J(words), J(idx), got}));
    }
    return list(out);
}

std::string fixture_record(const std::string& fn, const Value& v) {
    const std::string head = "[" + J(fn) + ",";
    if (fn == "fixture_voc") {
        const std::string saved = bf::data_dir();
        bf::set_data_dir(fixtures_dir + "/data");
        const std::string got = guard([&] { return J(sorted(bf::vocab::voc(v[1].str, "pt-pt"))); });
        bf::set_data_dir(saved);
        return head + J(v[1].str) + "," + got + "]";
    }
    const std::string path = fixtures_dir + "/models/" + v[1].str;
    if (fn == "fixture_model") {
        return head + J(v[1].str) + "," + guard([&] {
                   const auto m = bf::model::SenseModel::load(path);
                   std::vector<std::string> words;
                   for (const auto& item : m->words.items()) words.push_back(item.first);
                   std::vector<std::string> config;
                   for (const auto& [k, raw] : m->config.items()) config.push_back(J(std::vector<std::string>{k, raw}));
                   return list({J(m->model_type), J(m->lang), J(words), J(json_repr(m->version)), list(config)});
               }) + "]";
    }
    const auto m = bf::model::SenseModel::load(path);
    if (fn == "fixture_word") {
        const std::string word = v[2].str;
        return head + J(v[1].str) + "," + J(word) + "," + J(m->route(word)) + "," + J(repr(m->margin_tau(word))) +
               "," + fixture_runs(*m, word, v[5]) + "]";
    }
    bf::set_sense_model(m);
    std::string got;
    if (fn == "fixture_guess") {
        const bf::Words words = v[2].strings();
        const long idx = static_cast<long>(v[3].integer);
        got = head + J(v[1].str) + "," + J(words) + "," + J(idx) + "," +
              guard([&] { return J(bf::guess_sense(words, idx)); }) + "]";
    } else {
        got = head + J(v[1].str) + "," + J(v[2].str) + "," +
              guard([&] { return J(bf::add_extra_diacritics(v[2].str)); }) + "]";
    }
    bf::set_sense_model(bf::model::SenseModel::load());
    return got;
}

std::string call_record(const Value& v) {
    const std::string fn = v[0].str;
    const std::string head = "[" + J(fn) + ",";
    if (fn.rfind("fixture_", 0) == 0) return fixture_record(fn, v);
    if (fn == "bytes_sets") {
        bf::Words words;
        for (const auto& h : v[1].strings()) words.push_back(from_hex(h));
        bf::WordSet cue_set;
        for (const auto& h : v[2].strings()) cue_set.insert(from_hex(h));
        const long idx = static_cast<long>(v[3].integer);
        bf::Dict<bf::WordSet> vocs;
        vocs["v"] = cue_set;
        return head + json_line::dump(v[1]) + "," + json_line::dump(v[2]) + "," + J(idx) + "," +
               J(bf::text::cue_score(words, idx, cue_set, 4)) + "," +
               reprs(bf::features::extract_features(words, idx, vocs)) + "]";
    }
    if (fn == "cue_repr_custom") {
        const std::string t = v[1].str, other = v[2].str;
        const bf::cues::Cue cue{t, other, opt(v[3])};
        const bf::cues::CueRule one{other, {cue}}, none{t, {}}, two{t, {cue, bf::cues::Cue{other, t, {}}}};
        return head + J(t) + "," + J(other) + "," + J(opt(v[3])) + "," +
               J(std::vector<std::string>{bf::cues::repr(cue), bf::cues::repr(one), bf::cues::repr(none),
                                          bf::cues::repr(two)})
                   .substr(1);
    }
    if (fn == "probe") {
        return head + J(v[1].str) + "," + json_line::dump(v[2]) + "," +
               guard([&] { return probe_result(v[1].str, v[2]); }) + "]";
    }
    if (fn == "bytes") {
        const std::string raw = from_hex(v[1].str);
        const bf::Words words = bf::tokenize(raw);
        std::vector<std::string> senses;
        for (size_t i = 0; i < words.size(); ++i) {
            senses.push_back(guard([&] { return J(bf::guess_sense(words, static_cast<long>(i))); }));
        }
        return head + J(v[1].str) + "," + J(words) + "," + J(bits(bf::proper_flags(raw))) + "," +
               guard([&] { return J(bf::add_extra_diacritics(raw)); }) + "," + J(bf::is_ambiguous(raw)) + "," +
               J(bf::text::fold(raw)) + "," + J(bf::text::strip_punct(raw)) + "," + list(senses) + "]";
    }
    auto str = [&](const std::string& s) { return J(s); };
    auto ints = [](int x) { return J(x); };
    auto strs = [](const std::vector<std::string>& x) { return J(x); };
    const auto& t = bf::data::tables();

    if (fn == "HOMOGRAPHS") return head + items(t.HOMOGRAPHS, [&](const auto& d) { return dict(d, str); }) + "]";
    if (fn == "SENSE_POS") return head + items(t.SENSE_POS, [&](const auto& d) { return dict(d, str); }) + "]";
    if (fn == "SENSE_POSES") return head + items(t.SENSE_POSES, [&](const auto& d) { return items(d, strs); }) + "]";
    if (fn == "POS_SENSES") return head + items(t.POS_SENSES, [&](const auto& d) { return dict(d, strs); }) + "]";
    if (fn == "BASE_SCORE") return head + items(t.BASE_SCORE, [&](const auto& d) { return dict(d, ints); }) + "]";
    if (fn == "VERBS_IPA" || fn == "NOUNS_IPA" || fn == "ADP_IPA" || fn == "ADJ_IPA" || fn == "DEFAULT_SENSE" ||
        fn == "DEFAULT_POS") {
        return head + items(data_dict(fn), str) + "]";
    }
    if (fn == "SENSE_CUES") {
        std::vector<std::string> parts;
        for (const auto& [w, rule] : bf::cues::SENSE_CUES().items()) {
            std::vector<std::string> cues;
            for (const auto& c : rule.cues) cues.push_back(list({J(c.sense), J(c.voc), J(c.requires_prev)}));
            parts.push_back(list({J(w), J(rule.default_sense), list(cues)}));
        }
        return head + list(parts) + "]";
    }
    if (fn == "FEATURE_CUES") {
        return head + items(bf::cues::FEATURE_CUES(), [](const auto& p) { return J(std::vector<std::string>{p.first, p.second}); }) + "]";
    }
    if (fn == "DIACRITIZED") {
        std::vector<std::string> parts;
        for (const auto& d : bf::DIACRITIZED()) parts.push_back(J(std::vector<std::string>{d.base, d.sense, d.form}));
        return head + list(parts) + "]";
    }
    if (fn == "STRUCTURAL_VOCS") return head + J(bf::features::STRUCTURAL_VOCS()) + "]";
    if (fn == "DEFAULT_LANG") return head + J(bf::vocab::DEFAULT_LANG) + "]";
    auto file_name = [](const std::string& path) { return path.substr(path.find_last_of('/') + 1); };
    if (fn == "NB_PATH") return head + J(file_name(bf::model::NB_PATH())) + "]";
    if (fn == "PERCEPTRON_PATH") return head + J(file_name(bf::model::PERCEPTRON_PATH())) + "]";
    if (fn == "VERSION") return head + J(bf::version::VERSION_STR) + "]";
    if (fn == "load_structural_vocs") {
        const std::string lang = v[1].str;
        return head + J(lang) + "," +
               items(bf::features::load_structural_vocs(lang), [](const auto& s) { return J(sorted(s)); }) + "]";
    }
    if (fn == "voc") {
        const std::string name = v[1].str, lang = v[2].str;
        return head + J(name) + "," + J(lang) + "," + guard([&] { return J(sorted(bf::vocab::voc(name, lang))); }) + "]";
    }
    if (fn == "folded") {
        return head + J(v[1].str) + "," + J(sorted(bf::text::folded(bf::vocab::voc(v[1].str)))) + "]";
    }
    if (fn == "model") {
        const auto m = model_named(v[1].str);
        return head + J(v[1].str) + "," + m->version + "," + J(m->model_type) + "," + J(m->lang) + "," +
               items(m->config, [](const std::string& raw) { return raw; }) + "]";
    }
    if (fn == "model_word") {
        const auto m = model_named(v[1].str);
        const std::string word = v[2].str;
        const auto& entry = m->words.at(word);
        std::vector<std::string> weights;
        for (const auto& sense : entry.senses) {
            const bf::model::Weights* w = entry.weights.find(sense);
            std::vector<std::string> pairs;
            if (w) {
                for (const auto& [f, x] : *w) pairs.push_back(J(std::vector<std::string>{f, repr(x)}));
            }
            weights.push_back(list({J(sense), J(static_cast<long>(w ? w->size() : 0)),
                                    J(json_line::crc32_of(list(pairs)))}));
        }
        return head + J(v[1].str) + "," + J(word) + "," + J(m->has(word)) + "," + J(m->route(word)) + "," +
               J(repr(m->margin_tau(word))) + "," + J(entry.senses) + "," +
               items(entry.bias, [](double x) { return J(repr(x)); }) + "," + list(weights) + "]";
    }
    if (fn == "model_word_absent") {
        const auto m = model_named(v[1].str);
        const std::string word = v[2].str;
        return head + J(v[1].str) + "," + J(word) + "," + J(m->has(word)) + "," + J(m->route(word)) + "," +
               J(repr(m->margin_tau(word))) + "]";
    }
    auto str_items = [&](const bf::Dict<std::string>& d) { return items(d, str); };
    if (fn == "LECT_OVERRIDES") return head + items(t.LECT_OVERRIDES, str_items) + "]";
    if (fn == "lect_overrides") {
        const auto tag = opt(v[1]);
        return head + J(tag) + "," + guard([&] { return str_items(bf::lect_overrides(tag)); }) + "," +
               guard([&] { return str_items(bf::data::lect_overrides(tag)); }) + "]";
    }
    if (fn == "lect_diacritics") {
        const auto tag = opt(v[1]);
        const std::string s = v[2].str;
        return head + J(tag) + "," + J(s) + "," + guard([&] { return J(bf::add_extra_diacritics(s, tag)); }) + "]";
    }
    if (fn == "lect_alpha") {
        return head + J(v[1].str) + "," + J(!bf::data::lect_overrides(v[1].str).empty()) + "]";
    }
    if (fn == "infopedia.load") {
        return head + items(bf::infopedia::load(), [](const std::vector<bf::infopedia::Row>& rows) {
                   std::vector<std::string> parts;
                   for (const auto& row : rows) {
                       std::vector<std::string> cells;
                       for (const auto& [k, x] : row) cells.push_back(J(std::vector<std::string>{k, x}));
                       parts.push_back(list(cells));
                   }
                   return list(parts);
               }) + "]";
    }
    if (fn == "infopedia.readings_len") {
        return head + J(v[1].str) + "," + J(static_cast<long>(bf::infopedia::readings(v[1].str).size())) + "]";
    }
    if (fn == "infopedia.ipa") {
        std::optional<std::string> pos;
        if (!v[2].is_null()) pos = v[2].str;
        return head + J(v[1].str) + "," + J(pos) + "," + J(bf::infopedia::ipa(v[1].str, pos)) + "]";
    }
    if (fn == "unicode") {
        const std::string c = v[1].str, s = "a" + c + "b";
        return head + J(c) + "," + J(bf::tokenize(s)) + "," + J(bf::tokenize(c + "para")) + "," +
               J(bits(bf::proper_flags("x " + c + "y " + c))) + "," + J(bf::text::fold(s)) + "," +
               J(bf::text::strip_punct(s)) + "," + J(bf::is_ambiguous(c)) + "," +
               guard([&] { return J(bf::add_extra_diacritics(c + "Para " + c + "o carro")); }) + "]";
    }
    if (fn == "text") {
        const std::string s = v[1].str;
        return head + J(s) + "," + J(bf::text::strip_punct(s)) + "," + J(bf::text::fold(s)) + "," +
               J(bf::tokenize(s)) + "," + J(bits(bf::proper_flags(s))) + "," + J(bf::is_ambiguous(s)) + "]";
    }
    if (fn == "score_adp") {
        const bf::Words words = v[1].strings();
        const long idx = static_cast<long>(v[2].integer);
        return head + J(words) + "," + J(idx) + "," + guard([&] { return J(bf::scoring::score_adp(words, idx)); }) + "]";
    }
    if (fn == "cue_score") {
        const bf::Words words = v[1].strings();
        const long idx = static_cast<long>(v[2].integer), near = static_cast<long>(v[4].integer);
        const bf::WordSet cues = bf::text::folded(bf::vocab::voc(v[3].str));
        return head + J(words) + "," + J(idx) + "," + J(v[3].str) + "," + J(near) + "," +
               J(bf::text::cue_score(words, idx, cues, near)) + "]";
    }
    if (fn == "index") {
        const bf::Words words = v[1].strings();
        const long idx = static_cast<long>(v[2].integer);
        return head + J(words) + "," + J(idx) + "," + list({
                   guard([&] { return J(bf::guess_pos(words, idx)); }),
                   guard([&] { return J(bf::guess_sense(words, idx)); }),
                   guard([&] { return J(bf::guess_sense(words, idx, std::string("NOUN"))); }),
                   guard([&] { return J(bf::guess_sense(words, idx, std::string(""))); }),
                   guard([&] { return J(bf::disambiguate(words, idx)); }),
                   guard([&] { return J(bf::scoring::guess_pos(words, idx)); }),
                   four_scores(words, idx),
                   guard([&] { return feature_list(words, idx); }),
               }).substr(1);
    }
    if (fn == "disambiguate_sense") {
        const bf::Words words = v[1].strings();
        const long i = static_cast<long>(v[2].integer);
        const std::string sense = v[3].str;
        return head + J(words) + "," + J(i) + "," + J(sense) + "," +
               guard([&] { return J(bf::disambiguate(words, i, std::nullopt, sense)); }) + "]";
    }
    if (fn == "resolve_sense") {
        const std::string word = v[1].str, pos = v[4].str;
        const bf::Words words = v[2].strings();
        const long i = static_cast<long>(v[3].integer);
        return head + J(word) + "," + J(words) + "," + J(i) + "," + J(pos) + "," +
               guard([&] { return J(bf::scoring::resolve_sense(word, words, i, pos)); }) + "]";
    }
    if (fn.rfind("AMB", 0) == 0 || fn == "POS_AMBIGUOUS" || fn == "STOPPABLE_THINGS" || fn == "DET" ||
        fn == "QUANT" || fn == "PRON" || fn == "AUX_VERBS" || fn == "NUMERIC" || fn == "SOBRE_GOV" ||
        fn == "NEVER_AFTER_PREP" || fn == "PREP" || fn == "BEFORE_PREP" || fn == "AFTER_PREP") {
        return head + J(sorted(data_set(fn))) + "]";
    }
    return "unknown call " + fn;
}

struct Tally {
    long expected = 0, checked = 0, mismatches = 0;
};

// by_kind: count the mismatches of a calls file per record kind.
Tally run(const std::string& path, const std::function<std::string(const std::string&)>& record,
          bool by_kind = false) {
    const auto lines = json_line::read_gz_lines(path);
    Tally t;
    std::map<std::string, long> kinds;
    if (lines.empty()) throw std::runtime_error("empty golden file " + path);
    t.expected = json_line::header_count(lines[0]);
    for (size_t i = 1; i < lines.size(); ++i) {
        std::string got;
        try {
            got = record(lines[i]);
        } catch (const std::exception& e) {
            got = std::string("exception: ") + e.what();
        }
        ++t.checked;
        if (got != lines[i] && by_kind) {
            const Value v = json_line::parse(lines[i]);
            ++kinds[v[0].str == "probe" ? "probe " + v[1].str : v[0].str];
        }
        if (got != lines[i] && ++t.mismatches <= 10) {
            std::cerr << "MISMATCH " << path << "\n  python: " << lines[i].substr(0, 2000)
                      << "\n  c++:    " << got.substr(0, 2000) << "\n";
        }
    }
    std::cout << path.substr(path.find_last_of('/') + 1) << ": " << t.expected << " in, " << t.checked
              << " checked, " << t.mismatches << " mismatches\n";
    for (const auto& [kind, n] : kinds) std::cout << "  " << kind << ": " << n << " mismatches\n";
    return t;
}

int detail_of(const Value& line) {
    if (line[5].kind == Value::Int) return 0;
    if (line[5].size() == 0) return 1;
    return line[5][0][12].kind == Value::Array ? 2 : 1;
}

// The lect calls with lect_overrides.csv replaced by lect_nfd.csv, whose forms
// are not in NFC. The tables are read once, from one directory, so the data
// directory is copied under work_dir with that one file replaced.
int run_lect_nfd(const std::string& dir, const std::string& work_dir) {
    namespace fs = std::filesystem;
    const fs::path data = fs::path(work_dir) / "lect_nfd_data";
    fs::remove_all(data);
    fs::copy(bf::data_dir(), data, fs::copy_options::recursive);
    fs::copy_file(dir + "/lect_nfd.csv", data / "lect_overrides.csv", fs::copy_options::overwrite_existing);
    bf::set_data_dir(data.string());
    const Tally t = run(dir + "/lect_nfd.jsonl.gz", [](const std::string& line) { return call_record(json_line::parse(line)); });
    fs::remove_all(data);
    return (t.checked == t.expected && t.mismatches == 0) ? 0 : 1;
}

// Lect tables that the loader rejects. Each record holds the whole text of
// lect_overrides.csv and the exception, with the message after the path of the
// table. A rejected table leaves the tables unbuilt, so the next record loads
// its own table.
int run_lect_reject(const std::string& dir, const std::string& work_dir) {
    namespace fs = std::filesystem;
    const fs::path data = fs::path(work_dir) / "lect_reject_data";
    fs::remove_all(data);
    fs::copy(bf::data_dir(), data, fs::copy_options::recursive);
    bf::set_data_dir(data.string());
    const std::string path = (data / "lect_overrides.csv").string();
    const Tally t = run(dir + "/lect_reject.jsonl.gz", [&](const std::string& line) {
        const std::string table = json_line::parse(line)[1].str;
        std::ofstream(path, std::ios::binary) << table;
        std::string error;
        try {
            bf::data::lect_overrides(std::string("pt-BR"));
            error = "\"loaded\"";
        } catch (const bf::ValueError& e) {
            const std::string msg = e.what();
            const bool at_path = msg.compare(0, path.size(), path) == 0;
            error = "{\"err\":\"ValueError\",\"msg\":" + J(at_path ? msg.substr(path.size()) : msg) + "}";
        }
        return "[\"lect_reject\"," + J(table) + "," + error + "]";
    });
    fs::remove_all(data);
    return (t.checked == t.expected && t.mismatches == 0) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--lect-nfd") return run_lect_nfd(argv[2], argv[3]);
    if (argc == 4 && std::string(argv[1]) == "--lect-reject") return run_lect_reject(argv[2], argv[3]);
    if (argc != 2) {
        std::cerr << "usage: bifonia_parity <goldens dir>\n"
                     "       bifonia_parity --lect-nfd <goldens dir> <work dir>\n"
                     "       bifonia_parity --lect-reject <goldens dir> <work dir>\n";
        return 2;
    }
    const std::string dir = argv[1];
    fixtures_dir = dir + "/../fixtures";
    Tally total;
    bool counts_match = true;
    auto add = [&](const Tally& t) {
        total.expected += t.expected;
        total.checked += t.checked;
        total.mismatches += t.mismatches;
        // Each file must hold the count its header states, so a short file
        // cannot hide behind a long one.
        counts_match = counts_match && t.checked == t.expected;
    };
    add(run(dir + "/sentences.jsonl.gz", [](const std::string& line) {
        const Value v = json_line::parse(line);
        return sentence_record(v[0].str, detail_of(v));
    }));
    add(run(dir + "/calls.jsonl.gz", [](const std::string& line) { return call_record(json_line::parse(line)); }, true));

    // Route every model word to the model, so the model branch of guess_sense decides.
    auto forced = std::make_shared<bf::model::SenseModel>(*bf::model::SenseModel::load(bf::model::PERCEPTRON_PATH()));
    long k = 0;
    for (auto& item : forced->words.items()) {
        item.second.route = "model";
        item.second.margin_tau = static_cast<double>(k++ % 3) * 0.75;
    }
    bf::set_sense_model(forced);
    add(run(dir + "/forced.jsonl.gz", [](const std::string& line) {
        return forced_record(json_line::parse(line)[0].str);
    }));

    std::cout << "goldens: " << total.expected << " in, " << total.checked << " checked, " << total.mismatches
              << " mismatches\n";
    return (counts_match && total.mismatches == 0) ? 0 : 1;
}
