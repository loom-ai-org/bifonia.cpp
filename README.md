# bifonia.cpp

A C++17 port of [bifonia](https://github.com/TigreGotico/bifonia), which
disambiguates Portuguese heterophonic homographs. These are words such as
*acordo*, *sede* and *para*, whose spelling does not show the vowel quality. The
library picks the reading from the sentence context and gives its IPA. It can
also mark the reading with a diacritic that a rule-based G2P reads. The output is
the same as the Python package, byte for byte: see [PARITY.md](PARITY.md).

```cpp
#include <bifonia/bifonia.hpp>

bifonia::add_extra_diacritics("O autocarro para em frente ao hospital.");
// "O autocarro pára em frente ao hospital."
bifonia::add_extra_diacritics("Ele soma os números.", "pt-BR");
// "Ele sôma os números." (Brazilian Portuguese closes the vowel)

const auto words = bifonia::tokenize("Tinha tanta sede que bebi a garrafa toda.");
bifonia::guess_sense(words, 2);   // "thirst"
bifonia::disambiguate(words, 2);  // "ˈsedɨ"
bifonia::HOMOGRAPHS().contains("sede");  // true
```

The engine works in layers:

- Context scorers give each part of speech a score from the words around the
  homograph.
- Resolvers pick the meaning when two readings share a part of speech, for
  example *sede* (thirst or seat).
- Per-word sense models (naive Bayes and an averaged perceptron) can take over
  from the rules for a word. The shipped models route every word to the rules.

Each source file under `src/` matches one Python module.

## Data files

The library reads its data at run time: the homograph table, the word lists
under `locale/`, the infopedia table and the two sense models. `cmake --install`
puts them in `<prefix>/share/bifonia`, and the library looks there by default.
The library finds that directory from the path of the shared library, or of the
program that links the static library, so an install that
`cmake --install --prefix` puts in another prefix finds its own data. When that
directory does not hold the data, the library uses the install location that
the build compiles in. To use another directory, set the environment variable `BIFONIA_DATA_DIR` or
call `bifonia::set_data_dir()` before the first call. Each file is read once and
then cached, so repeated calls are cheap.

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
cmake --install build --prefix /usr/local
```

The tests need zlib, which reads the compressed goldens. The library itself needs
nothing.

To use the library from another CMake project:

```cmake
find_package(bifonia REQUIRED)
target_link_libraries(app PRIVATE bifonia::bifonia)
```

## Licence

Apache-2.0, the same licence as the Python package. See [LICENSE](LICENSE).
