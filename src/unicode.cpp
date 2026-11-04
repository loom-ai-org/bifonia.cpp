#include <dlfcn.h>
#if defined(__linux__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <sstream>

#include "detail.hpp"

#ifndef BIFONIA_DEFAULT_DATA_DIR
#define BIFONIA_DEFAULT_DATA_DIR "data"
#endif
#ifndef BIFONIA_DATA_FROM_LIBDIR
#define BIFONIA_DATA_FROM_LIBDIR "../share/bifonia"
#endif
#ifndef BIFONIA_DATA_FROM_BINDIR
#define BIFONIA_DATA_FROM_BINDIR "../share/bifonia"
#endif

namespace bifonia {

namespace {
std::mutex& data_dir_mutex() {
    static std::mutex m;
    return m;
}
std::optional<std::string>& data_dir_override() {
    static std::optional<std::string> value;
    return value;
}

// The installed data directory beside the shared library or the executable
// that holds this code, so that an install moved to another prefix finds its
// data. Empty when that directory does not hold the data.
std::string relocated_data_dir() {
    namespace fs = std::filesystem;
    Dl_info self{};
    if (dladdr(reinterpret_cast<void*>(&data_dir), &self) == 0 || self.dli_fname == nullptr) return {};
    fs::path object = self.dli_fname;
    // A shared library finds the data from LIBDIR, the main program from BINDIR.
    const char* relative = BIFONIA_DATA_FROM_LIBDIR;
#if defined(__linux__)
    // glibc names the main program by argv[0], which need not be a path.
    Dl_info program{};
    if (dladdr(reinterpret_cast<void*>(getauxval(AT_PHDR)), &program) != 0 &&
        program.dli_fbase == self.dli_fbase) {
        object = "/proc/self/exe";
        relative = BIFONIA_DATA_FROM_BINDIR;
    }
#elif defined(__APPLE__)
    if (self.dli_fbase == static_cast<const void*>(_dyld_get_image_header(0))) relative = BIFONIA_DATA_FROM_BINDIR;
#else
    return {};
#endif
    std::error_code ec;
    const fs::path candidate = (fs::canonical(object, ec).parent_path() / relative).lexically_normal();
    if (ec || !fs::is_regular_file(candidate / "heterophonic_homographs.csv", ec)) return {};
    return candidate.string();
}
}  // namespace

void set_data_dir(const std::string& path) {
    std::lock_guard<std::mutex> lock(data_dir_mutex());
    data_dir_override() = path;
}

std::string data_dir() {
    {
        std::lock_guard<std::mutex> lock(data_dir_mutex());
        if (data_dir_override()) return *data_dir_override();
    }
    if (const char* env = std::getenv("BIFONIA_DATA_DIR"); env != nullptr && *env != '\0') return env;
    static const std::string relocated = relocated_data_dir();
    return relocated.empty() ? BIFONIA_DEFAULT_DATA_DIR : relocated;
}

}  // namespace bifonia

namespace bifonia::detail {

namespace {
#include "unicode_tables.inc"

template <size_t N>
bool in_ranges(const char32_t (&table)[N][2], char32_t c) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (c < table[mid][0]) {
            hi = mid;
        } else if (c > table[mid][1]) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

template <size_t N>
const CaseEntry* find_case(const CaseEntry (&table)[N], char32_t c) {
    const auto* end = std::end(table);
    const auto* it = std::lower_bound(std::begin(table), end, c,
                                      [](const CaseEntry& e, char32_t v) { return e.cp < v; });
    return (it != end && it->cp == c) ? it : nullptr;
}

constexpr char32_t kCapitalSigma = 0x03A3;

// Python's handle_capital_sigma: Σ lowercases to final ς when it follows a
// cased letter (skipping case-ignorables) and no cased letter follows it.
char32_t lower_sigma(const U32& text, size_t i) {
    size_t j = i;
    char32_t c = 0;
    bool found = false;
    while (j > 0) {
        c = text[--j];
        if (!in_ranges(kCaseIgnorable, c)) {
            found = true;
            break;
        }
    }
    bool final_sigma = found && in_ranges(kCased, c);
    if (final_sigma && i + 1 < text.size()) {
        size_t k = i + 1;
        for (; k < text.size(); ++k) {
            c = text[k];
            if (!in_ranges(kCaseIgnorable, c)) break;
        }
        final_sigma = k == text.size() || !in_ranges(kCased, c);
    }
    return final_sigma ? 0x03C2 : 0x03C3;
}

int decimal_value(char32_t c) {
    const auto* end = std::end(kDecimal);
    const auto* it = std::lower_bound(std::begin(kDecimal), end, c,
                                      [](const char32_t (&e)[2], char32_t v) { return e[0] < v; });
    return (it != end && (*it)[0] == c) ? static_cast<int>((*it)[1]) : -1;
}

unsigned char combining_class(char32_t c) {
    const auto* end = std::end(kCcc);
    const auto* it = std::lower_bound(std::begin(kCcc), end, c,
                                      [](const CccEntry& e, char32_t v) { return e.cp < v; });
    return (it != end && it->cp == c) ? it->ccc : 0;
}

constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kTCount = 28, kNCount = 21 * 28, kSCount = 19 * 21 * 28;

}  // namespace

// Python's bytes.decode("utf-8", "replace"): each maximal subpart of an
// ill-formed sequence becomes one U+FFFD, and the byte that breaks a sequence
// starts the next one.
U32 from_utf8(std::string_view text) {
    U32 out;
    out.reserve(text.size());
    const size_t n = text.size();
    auto byte = [&](size_t k) { return static_cast<unsigned char>(text[k]); };
    size_t i = 0;
    while (i < n) {
        const unsigned char b = byte(i);
        if (b < 0x80) {
            out.push_back(b);
            ++i;
            continue;
        }
        size_t len;
        char32_t cp;
        unsigned char lo = 0x80, hi = 0xBF;  // the range of the second byte
        if (b >= 0xC2 && b <= 0xDF) {
            len = 2;
            cp = b & 0x1F;
        } else if (b >= 0xE0 && b <= 0xEF) {
            len = 3;
            cp = b & 0x0F;
            if (b == 0xE0) lo = 0xA0;
            if (b == 0xED) hi = 0x9F;
        } else if (b >= 0xF0 && b <= 0xF4) {
            len = 4;
            cp = b & 0x07;
            if (b == 0xF0) lo = 0x90;
            if (b == 0xF4) hi = 0x8F;
        } else {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        size_t k = 1;
        for (; k < len && i + k < n; ++k) {
            const unsigned char c = byte(i + k);
            const unsigned char min = k == 1 ? lo : 0x80, max = k == 1 ? hi : 0xBF;
            if (c < min || c > max) break;
            cp = (cp << 6) | (c & 0x3F);
        }
        if (k == len) {
            out.push_back(cp);
        } else {
            out.push_back(0xFFFD);
        }
        i += k;
    }
    return out;
}

std::string to_utf8(std::u32string_view text) {
    std::string out;
    out.reserve(text.size());
    for (char32_t cp : text) {
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
    return out;
}

U32 py_lower(const U32& text) {
    U32 out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        char32_t c = text[i];
        if (c == kCapitalSigma) {
            out.push_back(lower_sigma(text, i));
        } else if (const CaseEntry* e = find_case(kLower, c)) {
            out.append(e->to, e->len);
        } else {
            out.push_back(c);
        }
    }
    return out;
}

U32 py_upper(const U32& text) {
    U32 out;
    out.reserve(text.size());
    for (char32_t c : text) {
        if (const CaseEntry* e = find_case(kUpper, c)) {
            out.append(e->to, e->len);
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::string py_lower(const std::string& text) {
    for (unsigned char c : text) {
        if (c >= 0x80 || (c >= 'A' && c <= 'Z')) return to_utf8(py_lower(from_utf8(text)));
    }
    return text;
}

bool is_space(char32_t c) { return in_ranges(kSpace, c); }
bool is_word(char32_t c) { return in_ranges(kWord, c); }
bool is_upper(char32_t c) { return in_ranges(kIsUpper, c); }
bool is_digit(char32_t c) { return in_ranges(kIsDigit, c); }
bool is_alpha(char32_t c) { return in_ranges(kIsAlpha, c); }
bool is_mn(char32_t c) { return in_ranges(kMn, c); }
bool is_printable(char32_t c) { return in_ranges(kPrintable, c); }

U32 nfd(const U32& text) {
    U32 out;
    out.reserve(text.size());
    for (char32_t c : text) {
        if (c >= kSBase && c < kSBase + kSCount) {
            char32_t s = c - kSBase;
            out.push_back(kLBase + s / kNCount);
            out.push_back(kVBase + (s % kNCount) / kTCount);
            if (s % kTCount != 0) out.push_back(kTBase + s % kTCount);
            continue;
        }
        const auto* end = std::end(kDecomp);
        const auto* it = std::lower_bound(std::begin(kDecomp), end, c,
                                          [](const DecompEntry& e, char32_t v) { return e.cp < v; });
        if (it != end && it->cp == c) {
            out.append(kDecompPool + it->offset, it->len);
        } else {
            out.push_back(c);
        }
    }
    // Canonical ordering: a stable sort of each run of non-starters by class.
    size_t i = 0;
    while (i < out.size()) {
        if (combining_class(out[i]) == 0) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < out.size() && combining_class(out[j]) != 0) ++j;
        std::stable_sort(out.begin() + static_cast<long>(i), out.begin() + static_cast<long>(j),
                         [](char32_t a, char32_t b) { return combining_class(a) < combining_class(b); });
        i = j;
    }
    return out;
}

bool py_isdecimal(const std::string& text) {
    const U32 u = from_utf8(text);
    return !u.empty() && std::all_of(u.begin(), u.end(), [](char32_t c) {
        return (c >= '0' && c <= '9') || decimal_value(c) >= 0;
    });
}

std::string py_strip(const std::string& text) {
    U32 u = from_utf8(text);
    size_t begin = 0, end = u.size();
    while (begin < end && is_space(u[begin])) ++begin;
    while (end > begin && is_space(u[end - 1])) --end;
    return to_utf8(std::u32string_view(u).substr(begin, end - begin));
}

std::string strip_chars(std::string_view text, std::string_view chars) {
    return lstrip_chars(rstrip_chars(text, chars), chars);
}

std::string lstrip_chars(std::string_view text, std::string_view chars) {
    size_t b = 0;
    while (b < text.size() && chars.find(text[b]) != std::string_view::npos) ++b;
    return std::string(text.substr(b));
}

std::string rstrip_chars(std::string_view text, std::string_view chars) {
    size_t e = text.size();
    while (e > 0 && chars.find(text[e - 1]) != std::string_view::npos) --e;
    return std::string(text.substr(0, e));
}

std::vector<std::string> splitlines(const std::string& text) {
    U32 u = from_utf8(text);
    std::vector<std::string> lines;
    size_t start = 0, i = 0;
    while (i < u.size()) {
        char32_t c = u[i];
        bool brk = c == '\n' || c == '\r' || c == 0x0B || c == 0x0C || c == 0x1C || c == 0x1D ||
                   c == 0x1E || c == 0x85 || c == 0x2028 || c == 0x2029;
        if (!brk) {
            ++i;
            continue;
        }
        lines.push_back(to_utf8(std::u32string_view(u).substr(start, i - start)));
        i += (c == '\r' && i + 1 < u.size() && u[i + 1] == '\n') ? 2 : 1;
        start = i;
    }
    if (start < u.size()) lines.push_back(to_utf8(std::u32string_view(u).substr(start)));
    return lines;
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool last_in(std::string_view text, std::string_view chars) {
    return !text.empty() && chars.find(text.back()) != std::string_view::npos;
}

size_t cp_len(const std::string& text) { return from_utf8(text).size(); }

const std::string& at(const Words& words, long idx) {
    const long n = static_cast<long>(words.size());
    const long i = idx < 0 ? idx + n : idx;
    if (i < 0 || i >= n) throw IndexError("list index out of range");
    return words[static_cast<size_t>(i)];
}

std::string data_path(const std::string& relative) {
    return (std::filesystem::path(data_dir()) / relative).string();
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    out = buffer.str();
    if (!valid_utf8(out)) throw UnicodeDecodeError("'utf-8' codec can't decode " + path);
    return true;
}

std::vector<std::vector<std::pair<std::string, std::string>>> read_csv(const std::string& raw) {
    // Universal newlines first, as Python's text mode does.
    std::string text;
    text.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\r') {
            text.push_back('\n');
            if (i + 1 < raw.size() && raw[i + 1] == '\n') ++i;
        } else {
            text.push_back(raw[i]);
        }
    }
    // The csv module's state machine for the default dialect, not strict.
    enum State { StartRecord, StartField, InField, InQuoted, QuoteInQuoted };
    std::vector<std::vector<std::string>> records;
    std::vector<std::string> record;
    std::string field;
    State state = StartRecord;
    auto save_field = [&] {
        record.push_back(field);
        field.clear();
    };
    auto end_record = [&] {
        records.push_back(record);
        record.clear();
    };
    for (char c : text) {
        switch (state) {
            case StartRecord:
                if (c == '\n') break;  // an empty line yields no row
                state = StartField;
                [[fallthrough]];
            case StartField:
                if (c == '"') {
                    state = InQuoted;
                } else if (c == ',') {
                    save_field();
                } else if (c == '\n') {
                    save_field();
                    end_record();
                    state = StartRecord;
                } else {
                    field.push_back(c);
                    state = InField;
                }
                break;
            case InField:
                if (c == ',') {
                    save_field();
                    state = StartField;
                } else if (c == '\n') {
                    save_field();
                    end_record();
                    state = StartRecord;
                } else {
                    field.push_back(c);
                }
                break;
            case InQuoted:
                if (c == '"') {
                    state = QuoteInQuoted;
                } else {
                    field.push_back(c);
                }
                break;
            case QuoteInQuoted:
                if (c == '"') {
                    field.push_back('"');
                    state = InQuoted;
                } else if (c == ',') {
                    save_field();
                    state = StartField;
                } else if (c == '\n') {
                    save_field();
                    end_record();
                    state = StartRecord;
                } else {
                    field.push_back(c);
                    state = InField;
                }
                break;
        }
    }
    if (state != StartRecord) {
        save_field();
        end_record();
    }

    std::vector<std::vector<std::pair<std::string, std::string>>> rows;
    if (records.empty()) return rows;
    const auto& header = records[0];
    for (size_t r = 1; r < records.size(); ++r) {
        std::vector<std::pair<std::string, std::string>> row;
        for (size_t k = 0; k < header.size(); ++k) {
            row.emplace_back(header[k], k < records[r].size() ? records[r][k] : std::string());
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

// A Python str holds no ill-formed UTF-8: the bytes stand for the string they
// decode to, with U+FFFD for each maximal subpart. Only a key that decodes to
// a U+FFFD can equal an ill-formed member, so only such a key needs the scan.
bool contains(const WordSet& set, const std::string& word) {
    if (set.count(word) != 0) return true;
    if (std::all_of(word.begin(), word.end(), [](unsigned char c) { return c < 0x80; })) return false;
    const std::string key = to_utf8(from_utf8(word));
    if (key.find("\xEF\xBF\xBD") == std::string::npos) return false;
    for (const auto& member : set) {
        if (to_utf8(from_utf8(member)) == key) return true;
    }
    return false;
}

bool valid_utf8(const std::string& text) { return to_utf8(from_utf8(text)) == text; }

}  // namespace bifonia::detail
