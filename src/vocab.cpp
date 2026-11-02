// bifonia/vocab.py: the locale-aware .voc word-list loader.
#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>

#include "detail.hpp"

namespace bifonia::vocab {

namespace fs = std::filesystem;

namespace {
// _lang_dir: an exact directory wins, else the first directory (sorted by
// name) that shares the primary subtag, else the exact path.
fs::path lang_dir(const std::string& lang) {
    std::string norm = detail::py_lower(lang);
    std::replace(norm.begin(), norm.end(), '_', '-');
    const fs::path root = fs::path(data_dir()) / "locale";
    std::error_code ec;
    if (fs::is_directory(root / norm, ec)) return root / norm;
    const std::string primary = norm.substr(0, norm.find('-'));
    std::vector<fs::path> dirs;
    for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        dirs.push_back(it->path());
    }
    std::sort(dirs.begin(), dirs.end());
    for (const auto& d : dirs) {
        const std::string name = d.filename().string();
        if (fs::is_directory(d, ec) && name.substr(0, name.find('-')) == primary) return d;
    }
    return root / norm;
}
}  // namespace
}  // namespace bifonia::vocab

namespace bifonia::detail {
WordSet parse_voc(const std::string& text) {
    WordSet terms;
    for (const auto& line : detail::splitlines(text)) {
        const std::string t = detail::py_strip(line);
        if (!t.empty() && t[0] != '#') terms.insert(detail::py_lower(t));
    }
    return terms;
}
}  // namespace bifonia::detail

namespace bifonia::vocab {

const WordSet& voc(const std::string& name, const std::string& lang) {
    static std::mutex mutex;
    static std::map<std::pair<std::string, std::string>, std::unique_ptr<WordSet>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto& slot = cache[{name, lang}];
    if (slot) return *slot;
    const fs::path path = lang_dir(lang) / (name + ".voc");
    std::string text;
    std::error_code ec;
    if (!fs::exists(path, ec) || !detail::read_file(path.string(), text)) {
        cache.erase({name, lang});
        throw FileNotFoundError("vocabulary file not found: " + path.string());
    }
    slot = std::make_unique<WordSet>(detail::parse_voc(text));
    return *slot;
}

}  // namespace bifonia::vocab
