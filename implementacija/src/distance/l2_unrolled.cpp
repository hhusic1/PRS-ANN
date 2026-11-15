#include "distance.hpp"

#include "bench.hpp"

namespace ann
{
    // Skalarna sa 4 nezavisna akumulatora, bez SIMD-a (kompajlira se sa
    // -fno-tree-vectorize, kao l2_scalar). Razlika u odnosu na l2_scalar je
    // SAMO broj akumulatora, pa poređenje izoluje efekat zavisnosti podataka.
    //
    // VTune (docs/vtune/nalazi-pocetno-stanje.md): u l2_scalar svako
    // sabiranje `sum += diff * diff` čeka prethodno, pa petlja traje tačno
    // koliko kašnjenje jednog FP sabiranja (2 takta po elementu). Sa četiri
    // zbira četiri sabiranja idu kroz cjevovod istovremeno.
    float l2_scalar4(const float* PERF_RESTRICT a, const float* PERF_RESTRICT b, std::size_t d)
    {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        std::size_t i = 0;
        PERF_NO_VECTORIZE
        for (; i + 4 <= d; i += 4)
        {
            const float d0 = a[i] - b[i];
            const float d1 = a[i + 1] - b[i + 1];
            const float d2 = a[i + 2] - b[i + 2];
            const float d3 = a[i + 3] - b[i + 3];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
        }
        float sum = (s0 + s1) + (s2 + s3);
        PERF_NO_VECTORIZE
        for (; i < d; ++i)
        {
            const float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
