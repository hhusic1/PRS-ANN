#pragma once

// Čitanje skupova podataka u formatu .fvecs / .ivecs (TEXMEX, SIFT10K/SIFT1M).
//
// Format: svaki vektor je zapisan kao
//     int32 d, pa d vrijednosti (float32 za .fvecs, int32 za .ivecs).
// Sve je little-endian, što odgovara i x86 i ARM (Apple M) procesorima.
//
// U memoriji su vektori u jednom ravnom nizu red po red (n × d), bez
// std::vector<std::vector<float>>: red i počinje na data + i * d, pa je
// skeniranje kontinuirano (dobro za keš i SIMD).

#include "simd_support.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ann
{
    struct FloatMatrix
    {
        std::size_t n = 0;          // broj vektora
        std::size_t d = 0;          // dimenzija
        simd::AlignedFloats values; // n × d, poravnato na 32 B

        const float* row(std::size_t i) const
        {
            return values.data() + i * d;
        }
    };

    // Za groundtruth (ID-evi susjeda). Nije u vrućoj petlji, pa običan vector.
    struct IntMatrix
    {
        std::size_t n = 0;
        std::size_t d = 0;
        std::vector<std::int32_t> values; // n × d

        const std::int32_t* row(std::size_t i) const
        {
            return values.data() + i * d;
        }
    };

    // max_rows == 0 znači "pročitaj sve"; inače se čita samo prvih max_rows
    // vektora (npr. podskup upita za brzu provjeru).
    // Baca std::runtime_error ako fajl ne postoji ili nije ispravnog formata.
    FloatMatrix read_fvecs(const std::filesystem::path& path, std::size_t max_rows = 0);
    IntMatrix read_ivecs(const std::filesystem::path& path, std::size_t max_rows = 0);
}
