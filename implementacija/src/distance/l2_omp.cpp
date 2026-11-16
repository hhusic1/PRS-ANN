#include "distance.hpp"

#include "bench.hpp"

namespace ann
{
    // `omp simd` verzija (SIMD faza): direktiva `reduction(+:sum)` eksplicitno
    // dozvoljava da se zbir računa po trakama vektorskog registra, tj. drugim
    // redoslijedom sabiranja, pa kompajler vektorizuje i bez -ffast-math.
    // Kompajlira se sa -fopenmp-simd: uključene su samo simd direktive, bez
    // OpenMP runtime-a (niti dolaze u OpenMP fazi).
    //
    // Za SIFT (cijeli brojevi 0..255) rezultat je i dalje bit-identičan l2_scalar.
    float l2_omp(const float* PERF_RESTRICT a, const float* PERF_RESTRICT b, std::size_t d)
    {
        float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
        for (std::size_t i = 0; i < d; ++i)
        {
            const float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
