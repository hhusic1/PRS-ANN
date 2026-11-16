#include "distance.hpp"

// AVX2 + FMA verzije udaljenosti. Kompajliraju se sa -mavx2 -mfma (samo na
// x86, CMakeLists.txt); poziva ih se samo ako cpu_has_avx2_fma() (ivf-bench).
// Obje koriste FMA i razlikuju se SAMO po broju akumulatora:
//  - l2_avx2_fma1: jedan vektorski akumulator. Svaki _mm256_fmadd_ps čeka
//    prethodni, pa je i ovdje lanac zavisnosti (kašnjenje FMA po koraku od 8).
//  - l2_avx2_fma4: četiri akumulatora, četiri lanca paralelno.
// Učitavanja su neporavnata (loadu): redovi baze jesu poravnati na 32 B, ali
// funkcija ne smije to pretpostavljati (npr. testovi sa std::vector).
//
// Tačnost: za SIFT (cijeli brojevi 0..255) su sve međusume tačne u floatu, pa
// je rezultat bit-identičan l2_scalar bez obzira na redoslijed i FMA.

#if defined(__x86_64__) || defined(__i386__)

#include <immintrin.h>

namespace ann
{
    namespace
    {
        // Zbir 8 traka registra (horizontalna suma).
        inline float horizontal_sum(__m256 v)
        {
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_movehdup_ps(s));
            return _mm_cvtss_f32(s);
        }

        inline __m256 fma_step(const float* a, const float* b, __m256 acc)
        {
            const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
            return _mm256_fmadd_ps(diff, diff, acc);
        }

        inline float scalar_tail(const float* a, const float* b, std::size_t i, std::size_t d, float sum)
        {
            for (; i < d; ++i)
            {
                const float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum;
        }
    }

    float l2_avx2_fma1(const float* a, const float* b, std::size_t d)
    {
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= d; i += 8)
            acc = fma_step(a + i, b + i, acc);
        return scalar_tail(a, b, i, d, horizontal_sum(acc));
    }

    float l2_avx2_fma4(const float* a, const float* b, std::size_t d)
    {
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 32 <= d; i += 32)
        {
            acc0 = fma_step(a + i, b + i, acc0);
            acc1 = fma_step(a + i + 8, b + i + 8, acc1);
            acc2 = fma_step(a + i + 16, b + i + 16, acc2);
            acc3 = fma_step(a + i + 24, b + i + 24, acc3);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        for (; i + 8 <= d; i += 8)
            acc = fma_step(a + i, b + i, acc);
        return scalar_tail(a, b, i, d, horizontal_sum(acc));
    }
}

#endif
