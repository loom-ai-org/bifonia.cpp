// The examples of the Python package's own tests, checked against the C++ API.
#include <iostream>
#include <string>

#include "bifonia/bifonia.hpp"

namespace bf = bifonia;

namespace {
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

long index_of(const bf::Words& words, const std::string& word) {
    for (size_t i = 0; i < words.size(); ++i) {
        if (words[i] == word || words[i].substr(0, word.size()) == word) return static_cast<long>(i);
    }
    return -1;
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
}  // namespace

int main() {
    check(contains(bf::add_extra_diacritics("O autocarro para em frente ao hospital."), "pára"), "para → pára");
    check(contains(bf::add_extra_diacritics("O gosto do vinho é excelente."), "gôsto"), "gosto → gôsto");
    check(contains(bf::add_extra_diacritics("Eu gosto de música clássica."), "gósto"), "gosto → gósto");
    check(contains(bf::add_extra_diacritics("O acordo foi assinado ontem."), "acôrdo"), "acordo → acôrdo");
    check(contains(bf::add_extra_diacritics("Eu acordo cedo todos os dias."), "acórdo"), "acordo → acórdo");
    check(bf::add_extra_diacritics("Acordo cedo.") == "Acórdo cedo.", "capital kept");
    check(bf::add_extra_diacritics("Eu domo o leão.") == "Eu dómo o leão.", "domo → dómo");
    check(bf::add_extra_diacritics("Eu domo o leão.", std::string("pt-BR")) == "Eu domo o leão.", "pt-BR domo unmarked");
    check(bf::add_extra_diacritics("Ele soma os números.", std::string("pt-UY-x-mvd")) == "Ele sôma os números.",
          "pt-UY sub-lect soma → sôma");

    const auto words = bf::tokenize("A sede da empresa fica em Lisboa.");
    const long idx = index_of(words, "sede");
    const auto feats = bf::features::extract_features(words, idx, bf::features::load_structural_vocs("pt-pt"));
    check(feats.find("R1=da") && *feats.find("R1=da") == 1.0, "R1=da feature");
    check(feats.contains("W=empresa"), "W=empresa feature");
    check(feats.find("L1∈determiners") && *feats.find("L1∈determiners") == 1.0, "L1∈determiners feature");
    check(bf::guess_sense(words, idx) == std::optional<std::string>("seat"), "sede → seat");

    const auto thirst = bf::tokenize("Tinha tanta sede que bebi a garrafa toda.");
    check(bf::guess_sense(thirst, index_of(thirst, "sede")) == std::optional<std::string>("thirst"), "sede → thirst");

    const auto jogo = bf::tokenize("Eu jogo às cartas todas as noites.");
    check(bf::guess_sense(jogo, index_of(jogo, "jogo"), std::string("VERB")) == std::optional<std::string>("play"),
          "jogo VERB → play");
    check(bf::disambiguate(bf::tokenize("eu gosto de ti"), 1) == bf::HOMOGRAPHS().at("gosto").at("like"),
          "disambiguate gosto");

    const auto m = bf::model::SenseModel::load(bf::model::PERCEPTRON_PATH());
    const std::string pred = m->predict("sede", words, idx);
    check(pred == "seat" || pred == "thirst" || pred == "legal_context", "model predicts a valid sense");

    if (failures == 0) std::cout << "examples: all passed\n";
    return failures == 0 ? 0 : 1;
}
