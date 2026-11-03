// bifonia/infopedia.py: the bundled infopedia IPA dataset (infopedia_pt.csv).
#include "detail.hpp"

namespace bifonia::infopedia {

namespace {
const std::string* field(const Row& row, const char* name) {
    for (const auto& [key, value] : row) {
        if (key == name) return &value;
    }
    return nullptr;
}
}  // namespace

const Dict<std::vector<Row>>& load() {
    static const Dict<std::vector<Row>> rows = [] {
        Dict<std::vector<Row>> out;
        std::string text;
        if (!detail::read_file(detail::data_path("infopedia_pt.csv"), text)) return out;
        for (auto& row : detail::read_csv(text)) {
            const std::string* word = field(row, "word");
            if (word == nullptr) throw KeyError("word");
            out[*word].push_back(std::move(row));
        }
        return out;
    }();
    return rows;
}

const std::vector<Row>& readings(const std::string& word) {
    static const std::vector<Row> kNone;
    const std::vector<Row>* rows = load().find(detail::py_lower(word));
    return rows ? *rows : kNone;
}

std::optional<std::string> ipa(const std::string& word, const std::optional<std::string>& pos) {
    std::vector<const Row*> rows;
    for (const Row& r : readings(word)) rows.push_back(&r);
    if (pos && !pos->empty()) {
        // A reading tagged with pos alone wins over one whose tags include it.
        std::vector<const Row*> exact, tagged;
        for (const Row* r : rows) {
            const std::string* p = field(*r, "pos");
            if (p == nullptr) throw KeyError("pos");
            if (*p == *pos) exact.push_back(r);
        }
        for (const Row* r : rows) {
            const std::string& tags = *field(*r, "pos");
            for (size_t start = 0;;) {
                const size_t bar = tags.find('|', start);
                if (tags.compare(start, bar - start, *pos) == 0) {
                    tagged.push_back(r);
                    break;
                }
                if (bar == std::string::npos) break;
                start = bar + 1;
            }
        }
        if (!exact.empty()) {
            rows = exact;
        } else if (!tagged.empty()) {
            rows = tagged;
        }
    }
    if (rows.empty()) return std::nullopt;
    const std::string* value = field(*rows[0], "ipa");
    if (value == nullptr) throw KeyError("ipa");
    return *value;
}

}  // namespace bifonia::infopedia
