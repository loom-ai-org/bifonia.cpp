// Built against an installed bifonia to prove the CMake package and the
// installed data directory work. An argument names the data directory that
// data_dir() must find.
#include <bifonia/bifonia.hpp>
#include <filesystem>

int main(int argc, char** argv) {
    const bool ok = bifonia::add_extra_diacritics("O autocarro para em frente ao hospital.") ==
                        "O autocarro pára em frente ao hospital." &&
                    bifonia::HOMOGRAPHS().contains("sede");
    const bool found = argc < 2 || std::filesystem::equivalent(bifonia::data_dir(), argv[1]);
    return ok && found ? 0 : 1;
}
