// Internal helpers shared by the modules: the code-point layer that mirrors
// Python's str methods, list indexing with Python's rules, and file access.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "bifonia/bifonia.hpp"

namespace bifonia::detail {

using U32 = std::u32string;

// UTF-8 conversion. An invalid byte decodes to U+FFFD.
U32 from_utf8(std::string_view text);
std::string to_utf8(std::u32string_view text);

// Python str.lower() and str.upper(): full mappings, Greek final sigma included.
U32 py_lower(const U32& text);
U32 py_upper(const U32& text);
std::string py_lower(const std::string& text);

bool is_space(char32_t c);   // str.isspace()
bool is_word(char32_t c);    // re's \w: str.isalnum() or '_'
bool is_upper(char32_t c);   // str.isupper() on one code point
bool is_digit(char32_t c);   // str.isdigit() on one code point
bool is_alpha(char32_t c);   // str.isalpha() on one code point
bool is_mn(char32_t c);      // unicodedata.category(c) == "Mn"
bool is_printable(char32_t c);  // str.isprintable() on one code point
U32 nfd(const U32& text);    // unicodedata.normalize("NFD", text)
U32 nfc(const U32& text);    // unicodedata.normalize("NFC", text)
bool py_isupper(const U32& text);  // str.isupper()

// Python str.isdecimal(): not empty, and every code point is a decimal digit.
bool py_isdecimal(const std::string& text);

// Python str.strip() with no argument.
std::string py_strip(const std::string& text);
// Python str.strip(chars), lstrip(chars) and rstrip(chars) for ASCII chars.
std::string strip_chars(std::string_view text, std::string_view chars);
std::string lstrip_chars(std::string_view text, std::string_view chars);
std::string rstrip_chars(std::string_view text, std::string_view chars);
// Python str.splitlines().
std::vector<std::string> splitlines(const std::string& text);

bool ends_with(std::string_view text, std::string_view suffix);
bool starts_with(std::string_view text, std::string_view prefix);
// The last code point is one of the ASCII characters in chars.
bool last_in(std::string_view text, std::string_view chars);
size_t cp_len(const std::string& text);

// Python list indexing: negative counts from the end; out of range throws IndexError.
const std::string& at(const Words& words, long idx);

// Files under the data directory.
std::string data_path(const std::string& relative);
// Read a data file; false when it cannot be opened. Throws UnicodeDecodeError
// when it is not UTF-8, as Python's open(path, encoding="utf-8") does.
bool read_file(const std::string& path, std::string& out);

// Python's csv.DictReader over a file opened in text mode: universal newlines,
// the default dialect, empty rows skipped. Each row holds every column of the
// header; missing values are empty and extra values are dropped.
std::vector<std::vector<std::pair<std::string, std::string>>> read_csv(const std::string& text);

// Python's `word in set`, with ill-formed UTF-8 read as Python decodes it.
bool contains(const WordSet& set, const std::string& word);
bool valid_utf8(const std::string& text);

// The terms of a .voc file: stripped, lowercased, blank and '#' lines skipped.
WordSet parse_voc(const std::string& text);

// text.folded(vocab.voc(name)), computed once per name.
const WordSet& folded_voc(const std::string& name);

}  // namespace bifonia::detail
