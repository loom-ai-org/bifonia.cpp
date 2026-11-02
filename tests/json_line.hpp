// Just enough JSON to read a golden line and to write a value back the way
// Python's json.dumps(value, ensure_ascii=False, separators=(",", ":")) does,
// byte for byte.
#pragma once

#include <charconv>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <zlib.h>

namespace json_line {

inline std::string quote(const std::string& text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out + "\"";
}

inline void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// A parsed JSON value. Numbers are integers: the goldens write every float
// as its Python repr string.
struct Value {
    enum Kind { Null, Bool, Int, String, Array, Object } kind = Null;
    bool boolean = false;
    long long integer = 0;
    std::string str;
    std::vector<Value> items;
    std::vector<std::pair<std::string, Value>> members;

    const Value& operator[](size_t i) const { return items.at(i); }
    size_t size() const { return items.size(); }
    bool is_null() const { return kind == Null; }
    std::vector<std::string> strings() const {
        std::vector<std::string> out;
        for (const auto& v : items) out.push_back(v.str);
        return out;
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    Value parse() {
        Value v = value();
        if (p_ != s_.size()) throw std::runtime_error("extra data in golden line");
        return v;
    }

private:
    Value value() {
        Value v;
        const char c = s_.at(p_);
        if (c == '"') {
            v.kind = Value::String;
            v.str = string();
        } else if (c == '[') {
            v.kind = Value::Array;
            ++p_;
            if (s_.at(p_) == ']') {
                ++p_;
                return v;
            }
            while (true) {
                v.items.push_back(value());
                if (s_.at(p_++) == ']') break;
            }
        } else if (c == '{') {
            v.kind = Value::Object;
            ++p_;
            if (s_.at(p_) == '}') {
                ++p_;
                return v;
            }
            while (true) {
                std::string key = string();
                ++p_;  // ':'
                v.members.emplace_back(std::move(key), value());
                if (s_.at(p_++) == '}') break;
            }
        } else if (s_.compare(p_, 4, "null") == 0) {
            p_ += 4;
        } else if (s_.compare(p_, 4, "true") == 0) {
            v.kind = Value::Bool;
            v.boolean = true;
            p_ += 4;
        } else if (s_.compare(p_, 5, "false") == 0) {
            v.kind = Value::Bool;
            p_ += 5;
        } else {
            v.kind = Value::Int;
            const auto res = std::from_chars(s_.data() + p_, s_.data() + s_.size(), v.integer);
            if (res.ec != std::errc()) throw std::runtime_error("bad number in golden line");
            p_ = static_cast<size_t>(res.ptr - s_.data());
        }
        return v;
    }

    std::string string() {
        std::string out;
        ++p_;
        auto hex4 = [&](size_t at) { return static_cast<unsigned>(std::stoul(s_.substr(at, 4), nullptr, 16)); };
        while (s_.at(p_) != '"') {
            char c = s_[p_++];
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            char e = s_.at(p_++);
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
                    unsigned cp = hex4(p_);
                    p_ += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && s_.compare(p_, 2, "\\u") == 0) {
                        unsigned low = hex4(p_ + 2);
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        p_ += 6;
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    throw std::runtime_error("bad escape");
            }
        }
        ++p_;
        return out;
    }

    const std::string& s_;
    size_t p_ = 0;
};

inline Value parse(const std::string& line) { return Parser(line).parse(); }

// Write a value back exactly as json.dumps would.
inline std::string dump(const Value& v) {
    switch (v.kind) {
        case Value::Null: return "null";
        case Value::Bool: return v.boolean ? "true" : "false";
        case Value::Int: return std::to_string(v.integer);
        case Value::String: return quote(v.str);
        case Value::Array: {
            std::string out = "[";
            for (size_t i = 0; i < v.items.size(); ++i) out += (i ? "," : "") + dump(v.items[i]);
            return out + "]";
        }
        case Value::Object: {
            std::string out = "{";
            for (size_t i = 0; i < v.members.size(); ++i) {
                out += (i ? "," : "") + quote(v.members[i].first) + ":" + dump(v.members[i].second);
            }
            return out + "}";
        }
    }
    return "";
}

// Read every line of a gzip-compressed text file.
inline std::vector<std::string> read_gz_lines(const std::string& path) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (file == nullptr) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> lines;
    std::string current;
    char buffer[1 << 16];
    int n;
    while ((n = gzread(file, buffer, sizeof buffer)) > 0) {
        for (int i = 0; i < n; ++i) {
            if (buffer[i] == '\n') {
                lines.push_back(std::move(current));
                current.clear();
            } else {
                current.push_back(buffer[i]);
            }
        }
    }
    gzclose(file);
    if (!current.empty()) lines.push_back(std::move(current));
    return lines;
}

// The "count" of a golden header line such as {"...","count":87091}.
inline long header_count(const std::string& header) {
    size_t at = header.rfind("\"count\":");
    if (at == std::string::npos) throw std::runtime_error("golden header has no count");
    return std::stol(header.substr(at + 8));
}

inline unsigned long crc32_of(const std::string& text) {
    return crc32(0L, reinterpret_cast<const Bytef*>(text.data()), static_cast<uInt>(text.size()));
}

}  // namespace json_line
