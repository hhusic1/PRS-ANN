#include "distance.hpp"

#include "bench.hpp"

namespace ann
{
    // Auto-vektorizovana verzija (SIMD faza): ista petlja kao l2_scalar, ali BEZ
    // -fno-tree-vectorize, pa kompajler smije sam vektorizovati (-O3 -march=native).
    //
    // Bez -ffast-math kompajler ne smije promijeniti redoslijed float sabiranja,
    // a vektorizacija redukcije `sum += ...` upravo to radi (zbir po trakama). Zato
    // se ova petlja očekivano NE vektorizuje; izvještaj kompajlera
    // (-fopt-info-vec-missed) i mjerenje to provjeravaju (kao dot_auto u Lab 1).
    float l2_auto(const float* PERF_RESTRICT a, const float* PERF_RESTRICT b, std::size_t d)
    {
        float sum = 0.0f;
        for (std::size_t i = 0; i < d; ++i)
        {
            const float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
