// Built against an installed bifonia to prove the CMake package and the
// installed data directory work.
#include <bifonia/bifonia.hpp>

int main() {
    const bool ok = bifonia::add_extra_diacritics("O autocarro para em frente ao hospital.") ==
                        "O autocarro pára em frente ao hospital." &&
                    bifonia::HOMOGRAPHS().contains("sede");
    return ok ? 0 : 1;
}
