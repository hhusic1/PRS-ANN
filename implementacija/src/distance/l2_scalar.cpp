#include "distance.hpp"

#include "bench.hpp"

namespace ann
{
    // Skalarna referenca. Kompajlira se sa -fno-tree-vectorize (CMakeLists.txt),
    // a PERF_NO_VECTORIZE (GCC >= 14 / clang) to dodatno osigurava na nivou
    // petlje, da i Release build ostane pravo skalarno poređenje za SIMD fazu.
    //
    // Napomena o tačnosti: SIFT vrijednosti su cijeli brojevi 0..255 zapisani
    // kao float. Razlike su cijeli brojevi, kvadrati <= 255^2, a zbir za
    // d = 128 je <= 128 * 255^2 = 8.323.200 < 2^24. Do 2^24 float tačno
    // predstavlja svaki cijeli broj, pa je ova udaljenost TAČNA (bez greške
    // zaokruživanja), bez obzira na redoslijed sabiranja.
    float l2_scalar(const float* PERF_RESTRICT a, const float* PERF_RESTRICT b, std::size_t d)
    {
        float sum = 0.0f;
        PERF_NO_VECTORIZE
        for (std::size_t i = 0; i < d; ++i)
        {
            const float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
