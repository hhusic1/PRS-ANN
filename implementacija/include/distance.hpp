#pragma once

// Funkcije udaljenosti. Sve vraćaju KVADRAT L2 udaljenosti (bez korijena):
// korijen je monotona funkcija, pa ne mijenja poredak susjeda, a košta.
//
// Svaka verzija je u svom .cpp fajlu jer dobija svoje zastavice kompajlera
// (kao dot_* u Lab 1). IVF kod prima funkciju kao parametar tipa DistFn, pa
// isti kod radi sa skalarnom, auto, omp i AVX2 verzijom.

#include <cstddef>
#include <string_view>
#include <vector>

namespace ann
{
    using DistFn = float (*)(const float*, const float*, std::size_t);

    // Referentna verzija: obična petlja, kompajlira se sa -fno-tree-vectorize.
    float l2_scalar(const float* a, const float* b, std::size_t d);

    // Optimizacije iz VTune analize (docs/vtune/). Razlikuju se samo po onome
    // što se mjeri: broj akumulatora (zavisnost podataka) i širina (SIMD).
    float l2_scalar4(const float* a, const float* b, std::size_t d);   // skalarna, 4 akumulatora
    float l2_avx2_fma1(const float* a, const float* b, std::size_t d); // AVX2 + FMA, 1 akumulator (samo x86)
    float l2_avx2_fma4(const float* a, const float* b, std::size_t d); // AVX2 + FMA, 4 akumulatora (samo x86)

    // SIMD faza: ista petlja kao l2_scalar, ali sa dozvoljenom vektorizacijom.
    float l2_auto(const float* a, const float* b, std::size_t d);      // kompajler sam (bez -ffast-math)
    float l2_omp(const float* a, const float* b, std::size_t d);       // #pragma omp simd reduction

    // Varijanta udaljenosti sa imenom, za izbor iz komandne linije (ivf-bench).
    struct DistanceVariant
    {
        const char* name;
        DistFn fn;
        bool needs_avx2_fma; // traži procesor sa AVX2 i FMA (provjera: cpu_has_avx2_fma)
    };

    // Sve varijante; prva je skalarna referenca.
    const std::vector<DistanceVariant>& distance_variants();

    // Varijanta po imenu, ili nullptr ako ne postoji.
    const DistanceVariant* find_distance(std::string_view name);

    // Da li procesor na kome program radi podržava AVX2 i FMA. Provjera je u
    // vrijeme izvršavanja, da AVX2 varijanta na starijem procesoru da jasnu
    // grešku umjesto pada na nepodržanoj instrukciji.
    bool cpu_has_avx2_fma();
}
