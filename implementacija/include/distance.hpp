#pragma once

// Funkcije udaljenosti. Sve vraćaju KVADRAT L2 udaljenosti (bez korijena):
// korijen je monotona funkcija, pa ne mijenja poredak susjeda, a košta.
//
// Svaka verzija je u svom .cpp fajlu jer dobija svoje zastavice kompajlera
// (kao dot_* u Lab 1). IVF kod prima funkciju kao parametar tipa DistFn, pa
// isti kod radi sa skalarnom, auto, omp i AVX2 verzijom.

#include <cstddef>

namespace ann
{
    using DistFn = float (*)(const float*, const float*, std::size_t);

    // Referentna verzija: obična petlja, kompajlira se sa -fno-tree-vectorize.
    float l2_scalar(const float* a, const float* b, std::size_t d);

    // SIMD faza: l2_auto, l2_omp, l2_avx2 (dodati ovdje).
}
