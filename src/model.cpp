// bifonia/model.py: inference for the per-word NB and perceptron sense models.
//
// The model files are JSON. A small reader made for their schema loads them,
// so the library needs no JSON dependency. Numbers are read as doubles; the
// shipped models hold only floats.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>

#include "detail.hpp"

namespace bifonia::model {

namespace {
using JsonError = JSONDecodeError;

class Reader {
public:
    explicit Reader(const std::string& text) : s_(text) {}

    void ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) ++p_;
    }
    char peek() {
        ws();
        if (p_ >= s_.size()) throw JsonError("unexpected end of JSON");
        return s_[p_];
    }
    void expect(char c) {
        if (peek() != c) throw JsonError(std::string("expected '") + c + "'");
        ++p_;
    }
    bool at_end() {
        ws();
        return p_ >= s_.size();
    }

    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            const size_t stop = s_.find_first_of("\"\\", p_);
            if (stop == std::string::npos) throw JsonError("unterminated string");
            out.append(s_, p_, stop - p_);
            p_ = stop + 1;
            if (s_[stop] == '"') return out;
            if (p_ >= s_.size()) throw JsonError("bad escape");
            const char e = s_[p_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    char32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && s_.compare(p_, 2, "\\u") == 0) {
                        const size_t back = p_;
                        p_ += 2;
                        const char32_t low = hex4();
                        if (low >= 0xDC00 && low < 0xE000) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        } else {
                            p_ = back;
                        }
                    }
                    // A lone surrogate has no UTF-8 form.
                    if (cp >= 0xD800 && cp < 0xE000) cp = 0xFFFD;
                    out += detail::to_utf8(std::u32string(1, cp));
                    break;
                }
                default:
                    throw JsonError("bad escape");
            }
        }
    }

    // A number, or NaN / Infinity / -Infinity as Python's json accepts them.
    double number(std::string* raw = nullptr) {
        ws();
        const size_t start = p_;
        for (const char* lit : {"NaN", "Infinity", "-Infinity"}) {
            if (s_.compare(p_, std::char_traits<char>::length(lit), lit) == 0) {
                p_ += std::char_traits<char>::length(lit);
                if (raw) *raw = lit;
                if (lit[0] == 'N') return std::numeric_limits<double>::quiet_NaN();
                return lit[0] == '-' ? -std::numeric_limits<double>::infinity()
                                     : std::numeric_limits<double>::infinity();
            }
        }
        while (p_ < s_.size() && std::string_view("+-0123456789.eE").find(s_[p_]) != std::string_view::npos) ++p_;
        double value = 0;
        const auto res = std::from_chars(s_.data() + start, s_.data() + p_, value);
        if (res.ec == std::errc::invalid_argument || res.ptr != s_.data() + p_) throw JsonError("bad number");
        if (raw) *raw = s_.substr(start, p_ - start);
        return value;
    }

    // Skip any value; returns its JSON text.
    std::string skip() {
        ws();
        const size_t start = p_;
        const char c = peek();
        if (c == '"') {
            string();
        } else if (c == '{' || c == '[') {
            const char close = c == '{' ? '}' : ']';
            ++p_;
            if (peek() == close) {
                ++p_;
            } else {
                while (true) {
                    if (c == '{') {
                        string();
                        expect(':');
                    }
                    skip();
                    if (peek() == ',') {
                        ++p_;
                        continue;
                    }
                    expect(close);
                    break;
                }
            }
        } else if (s_.compare(p_, 4, "true") == 0 || s_.compare(p_, 4, "null") == 0) {
            p_ += 4;
        } else if (s_.compare(p_, 5, "false") == 0) {
            p_ += 5;
        } else {
            number();
        }
        return s_.substr(start, p_ - start);
    }

    // Calls f(key) for each member; f must consume the value.
    template <class F>
    void object(F&& f) {
        expect('{');
        if (peek() == '}') {
            ++p_;
            return;
        }
        while (true) {
            const std::string key = string();
            expect(':');
            f(key);
            if (peek() == ',') {
                ++p_;
                continue;
            }
            expect('}');
            return;
        }
    }

    template <class F>
    void array(F&& f) {
        expect('[');
        if (peek() == ']') {
            ++p_;
            return;
        }
        while (true) {
            f();
            if (peek() == ',') {
                ++p_;
                continue;
            }
            expect(']');
            return;
        }
    }

private:
    char32_t hex4() {
        if (p_ + 4 > s_.size()) throw JsonError("bad \\u escape");
        unsigned value = 0;
        const auto res = std::from_chars(s_.data() + p_, s_.data() + p_ + 4, value, 16);
        if (res.ptr != s_.data() + p_ + 4) throw JsonError("bad \\u escape");
        p_ += 4;
        return value;
    }

    const std::string& s_;
    size_t p_ = 0;
};

WordEntry read_entry(Reader& r) {
    WordEntry e;
    r.object([&](const std::string& key) {
        if (key == "route") {
            e.route = r.string();
        } else if (key == "margin_tau") {
            e.margin_tau = r.number();
        } else if (key == "senses") {
            e.senses.clear();
            r.array([&] { e.senses.push_back(r.string()); });
        } else if (key == "bias") {
            e.bias = Dict<double>();
            r.object([&](const std::string& sense) { e.bias[sense] = r.number(); });
        } else if (key == "weights") {
            e.weights = Dict<Weights>();
            r.object([&](const std::string& sense) {
                Weights w;
                r.object([&](const std::string& feature) { w.emplace_back(feature, r.number()); });
                // Sorted for lookup; a repeated key keeps its last value, as in Python.
                std::stable_sort(w.begin(), w.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                Weights unique;
                for (auto& item : w) {
                    if (!unique.empty() && unique.back().first == item.first) {
                        unique.back().second = item.second;
                    } else {
                        unique.push_back(std::move(item));
                    }
                }
                e.weights[sense] = std::move(unique);
            });
        } else {
            r.skip();
        }
    });
    return e;
}

std::shared_ptr<SenseModel> parse_model(const std::string& text) {
    auto m = std::make_shared<SenseModel>();
    Reader r(text);
    r.object([&](const std::string& key) {
        if (key == "version") {
            m->version = static_cast<long>(r.number());
        } else if (key == "model_type") {
            m->model_type = r.string();
        } else if (key == "config") {
            m->config = Dict<std::string>();
            m->lang = "pt-pt";
            r.object([&](const std::string& name) {
                if (name == "lang" && r.peek() == '"') {
                    const std::string lang = r.string();
                    m->lang = lang;
                    m->config[name] = "\"" + lang + "\"";
                } else {
                    m->config[name] = r.skip();
                }
            });
        } else if (key == "words") {
            m->words = Dict<WordEntry>();
            r.object([&](const std::string& word) { m->words[word] = read_entry(r); });
        } else {
            r.skip();
        }
    });
    if (!r.at_end()) throw JsonError("extra data after JSON");
    return m;
}
}  // namespace

std::string NB_PATH() { return detail::data_path("sense_model_nb.json"); }
std::string PERCEPTRON_PATH() { return detail::data_path("sense_model_perceptron.json"); }

const double* WordEntry::weight(const std::string& sense, const std::string& feature) const {
    const Weights* w = weights.find(sense);
    if (w == nullptr) return nullptr;
    auto it = std::lower_bound(w->begin(), w->end(), feature,
                               [](const std::pair<std::string, double>& a, const std::string& f) { return a.first < f; });
    return (it != w->end() && it->first == feature) ? &it->second : nullptr;
}

std::shared_ptr<const SenseModel> SenseModel::load(const std::optional<std::string>& path) {
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<const SenseModel>> cache;
    const std::string file = (path && !path->empty()) ? *path : PERCEPTRON_PATH();
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(file);
    if (it != cache.end()) return it->second;
    std::string text;
    if (!detail::read_file(file, text)) throw FileNotFoundError("No such file: " + file);
    auto m = parse_model(text);
    cache.emplace(file, m);
    return m;
}

bool SenseModel::has(const std::string& word) const { return words.contains(word); }

std::string SenseModel::route(const std::string& word) const {
    const WordEntry* e = words.find(word);
    return e ? e->route : "rules";
}

double SenseModel::margin_tau(const std::string& word) const {
    const WordEntry* e = words.find(word);
    return e ? e->margin_tau : 0.0;
}

Dict<double> SenseModel::scores(const std::string& word, const Words& tokens, long idx,
                                const Dict<WordSet>* vocs) const {
    const WordEntry& entry = words.at(word);
    if (vocs == nullptr) vocs = &features::load_structural_vocs(lang);
    const Dict<double> feats = features::extract_features(tokens, idx, *vocs);
    Dict<double> out;
    for (const auto& sense : entry.senses) {
        const double* b = entry.bias.find(sense);
        double s = b ? *b : 0.0;
        for (const auto& [f, v] : feats.items()) {
            const double* wf = entry.weight(sense, f);
            if (wf && *wf != 0.0) s += v * *wf;
        }
        out[sense] = s;
    }
    return out;
}

std::string SenseModel::predict(const std::string& word, const Words& tokens, long idx,
                                const Dict<WordSet>* vocs) const {
    const Dict<double> s = scores(word, tokens, idx, vocs);
    std::vector<std::string> names;
    for (const auto& item : s.items()) names.push_back(item.first);
    std::sort(names.begin(), names.end());
    if (names.empty()) throw ValueError("max() arg is an empty sequence");
    std::string best = names[0];
    for (const auto& name : names) {
        if (s.at(name) > s.at(best)) best = name;
    }
    return best;
}

double SenseModel::margin(const std::string& word, const Words& tokens, long idx,
                          const Dict<WordSet>* vocs) const {
    std::vector<double> values;
    const Dict<double> s = scores(word, tokens, idx, vocs);
    for (const auto& item : s.items()) values.push_back(item.second);
    if (values.size() < 2) return std::numeric_limits<double>::infinity();
    std::sort(values.begin(), values.end(), std::greater<double>());
    return values[0] - values[1];
}

}  // namespace bifonia::model
