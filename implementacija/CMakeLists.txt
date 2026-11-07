cmake_minimum_required(VERSION 3.20)
project(PRS_ANN VERSION 0.1 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Osnovna podešavanja
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  # MSVC podržava samo OpenMP 2.0 (nema `omp simd reduction`), vidi CLAUDE.md.
  message(FATAL_ERROR "MSVC nije podržan. Koristiti GCC iz MSYS2 UCRT64 (ili clang).")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Debug ili Release" FORCE)
endif()

# Debug: bez optimizacije, sa simbolima. Release: sva mjerenja.
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# -march=native je x86 zastavica; na ARM-u (npr. Apple M-čipovi) ekvivalent je
# -mcpu=native. Zato se bira prema arhitekturi, da projekat radi i na Macu.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ANN_CPU)
if(ANN_CPU MATCHES "^(x86_64|amd64|x64|i[3-6]86)$")
  set(ANN_IS_X86 ON)
  set(ANN_NATIVE_FLAG -march=native)
elseif(ANN_CPU MATCHES "^(arm64|aarch64)$")
  set(ANN_IS_X86 OFF)
  set(ANN_NATIVE_FLAG -mcpu=native)
else()
  set(ANN_IS_X86 OFF)
  set(ANN_NATIVE_FLAG "")
endif()

# ---------------------------------------------------------------------------
# Biblioteka sa svim verzijama (dijele je program i kasnije testovi/mjerenja)
# ---------------------------------------------------------------------------
add_library(ann_core STATIC
  src/common/bench.cpp
  src/common/dataset.cpp
  src/common/eval.cpp
  src/distance/l2_scalar.cpp
  src/seq/bruteforce_seq.cpp
  src/seq/kmeans_seq.cpp
  src/seq/ivf_seq.cpp
)
target_include_directories(ann_core PUBLIC include)
target_compile_options(ann_core PUBLIC
  -Wall -Wextra -Wpedantic
  # Zabrana spajanja a*b + c u FMA kod skalarnog koda. Apple clang to radi po
  # defaultu, GCC na x86 bez -mfma ne može, pa bi se udaljenosti razlikovale u
  # zadnjim bitima između računara. Sekvencijalna verzija je referenca i mora
  # davati iste rezultate svuda. (AVX2 intrinsici _mm256_fmadd_ps nisu
  # pogođeni: to je eksplicitna FMA instrukcija, ne spajanje kompajlera.)
  -ffp-contract=off
  $<$<CONFIG:Release>:${ANN_NATIVE_FLAG}>
)

# Skalarna referenca mora ostati skalarna, i u Release buildu (kao u Lab 1).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(src/distance/l2_scalar.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(src/distance/l2_scalar.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-vectorize;-fno-slp-vectorize")
endif()

# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------
add_executable(prs_ann src/main.cpp)
target_link_libraries(prs_ann PRIVATE ann_core)

# Testovi za granične slučajeve (prazni klasteri, jednake udaljenosti).
enable_testing()
add_executable(ann_tests tests/test_seq.cpp)
target_link_libraries(ann_tests PRIVATE ann_core)
add_test(NAME ann_tests COMMAND ann_tests)
