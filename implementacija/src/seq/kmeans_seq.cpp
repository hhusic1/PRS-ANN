#include "kmeans.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ann
{
    namespace
    {
        // Generator slučajnih brojeva: SplitMix64 (Steele, Lea, Flood 2014).
        //
        // Zašto ne std::uniform_int_distribution: standard ne propisuje njen
        // algoritam, pa GCC (libstdc++) i Apple clang (libc++) iz istog seeda
        // daju RAZLIČITE brojeve -> različiti centroidi na Windowsu i macOS-u.
        // Zašto ne next_random iz Lab 1: to je LCG modulo 2^64, a njegovi niži
        // bitovi imaju kratak period (najniži bit samo alternira 0, 1, 0, 1),
        // pa je `next_random % m` loš izbor indeksa. SplitMix64 je brz, ima
        // dobro izmiješane sve bitove i ovdje je potpuno definisan.
        std::uint64_t splitmix64(std::uint64_t& state)
        {
            std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        }

        // Dodjela svih tačaka najbližem centroidu. Vraća ciljnu funkciju.
        // Zbir se računa u double: 100.000 sabiraka reda veličine 10^4-10^5
        // bi u floatu izgubio cifre, a ovdje nam treba precizno praćenje opadanja.
        double assign_all(const FloatMatrix& data, const FloatMatrix& centroids, DistFn dist,
                          std::vector<std::int32_t>& labels, std::vector<float>& dists)
        {
            double objective = 0.0;
            for (std::size_t i = 0; i < data.n; ++i)
            {
                float best = 0.0f;
                labels[i] = static_cast<std::int32_t>(nearest_centroid(data.row(i), centroids, dist, &best));
                dists[i] = best;
                objective += best;
            }
            return objective;
        }
    }

    std::vector<std::size_t> kmeans_init_indices(std::size_t n, std::size_t nlist, std::uint64_t seed)
    {
        if (nlist == 0 || nlist > n)
            throw std::invalid_argument("k-means: nlist mora biti izmedju 1 i broja tacaka");

        // Djelimični Fisher-Yates: prvih nlist pozicija dobija nasumične,
        // međusobno RAZLIČITE indekse (bez ponavljanja iste tačke).
        // Pristrasnost operacije % je zanemariva: m <= 10^6, a 2^64 / m je
        // ogroman, pa je odstupanje vjerovatnoće reda 10^-13.
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), std::size_t{ 0 });
        std::uint64_t state = seed;
        for (std::size_t i = 0; i < nlist; ++i)
        {
            const std::size_t j = i + static_cast<std::size_t>(splitmix64(state) % (n - i));
            std::swap(idx[i], idx[j]);
        }
        idx.resize(nlist);
        return idx;
    }

    std::size_t nearest_centroid(const float* x, const FloatMatrix& centroids,
                                 DistFn dist, float* best_dist)
    {
        std::size_t best = 0;
        float best_d = std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < centroids.n; ++c)
        {
            const float d = dist(x, centroids.row(c), centroids.d);
            // Strogo "<": pri jednakoj udaljenosti ostaje manji indeks
            // (deterministično, isto kao tie-break u top-k).
            if (d < best_d)
            {
                best_d = d;
                best = c;
            }
        }
        if (best_dist)
            *best_dist = best_d;
        return best;
    }

    KMeansResult kmeans_train_seq(const FloatMatrix& data, const KMeansParams& params)
    {
        const std::size_t n = data.n;
        const std::size_t d = data.d;
        const std::size_t nlist = params.nlist;
        if (params.iterations < 1)
            throw std::invalid_argument("k-means: broj iteracija mora biti >= 1");

        KMeansResult result;
        result.centroids.n = nlist;
        result.centroids.d = d;
        result.centroids.values = simd::AlignedFloats(nlist * d);
        float* cent = result.centroids.values.data();

        // Inicijalizacija: centroidi su nlist nasumično izabranih tačaka.
        const std::vector<std::size_t> init = kmeans_init_indices(n, nlist, params.seed);
        for (std::size_t c = 0; c < nlist; ++c)
            std::memcpy(cent + c * d, data.row(init[c]), d * sizeof(float));

        std::vector<std::int32_t> labels(n);
        std::vector<float> dists(n);
        // Sume po klasterima u double: prosjek hiljada floatova u float
        // akumulatoru gubi preciznost, a i OpenMP verzija će sabirati u double.
        std::vector<double> sums(nlist * d);
        std::vector<std::size_t> counts(nlist);
        std::vector<std::size_t> order(n);

        for (int it = 0; it < params.iterations; ++it)
        {
            // 1. Dodjela.
            result.objective.push_back(assign_all(data, result.centroids, params.dist, labels, dists));

            // 2. Ažuriranje: centroid = prosjek tačaka klastera.
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(counts.begin(), counts.end(), std::size_t{ 0 });
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto c = static_cast<std::size_t>(labels[i]);
                ++counts[c];
                const float* x = data.row(i);
                double* s = sums.data() + c * d;
                for (std::size_t j = 0; j < d; ++j)
                    s[j] += x[j];
            }

            std::vector<std::size_t> empty;
            for (std::size_t c = 0; c < nlist; ++c)
            {
                if (counts[c] == 0)
                {
                    // Prazan klaster: dijeljenje nulom bi dalo NaN centroid.
                    empty.push_back(c);
                    continue;
                }
                const double inv = 1.0 / static_cast<double>(counts[c]);
                for (std::size_t j = 0; j < d; ++j)
                    cent[c * d + j] = static_cast<float>(sums[c * d + j] * inv);
            }

            // 3. Prazni klasteri: centroid se premješta na tačku koja je
            //    NAJDALJE od svog centroida (najlošije opisana tačka). Za više
            //    praznih klastera uzimaju se redom najdalje tačke (svaka samo
            //    jednom). Pri jednakoj udaljenosti manji indeks tačke.
            //
            //    Ciljna funkcija i dalje ne raste: prazan klaster ne doprinosi
            //    zbiru, pa premještanje ne mijenja vrijednost za trenutnu dodjelu,
            //    a u sljedećoj dodjeli ta tačka dobija udaljenost 0.
            //    (FAISS umjesto toga dijeli najveći klaster na dva.)
            if (!empty.empty())
            {
                std::iota(order.begin(), order.end(), std::size_t{ 0 });
                std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(empty.size()), order.end(),
                                  [&](std::size_t a, std::size_t b)
                                  {
                                      return dists[a] > dists[b] || (dists[a] == dists[b] && a < b);
                                  });
                for (std::size_t e = 0; e < empty.size(); ++e)
                    std::memcpy(cent + empty[e] * d, data.row(order[e]), d * sizeof(float));
            }
            result.relocated.push_back(empty.size());
        }

        // Konačna dodjela: ciljna funkcija nakon zadnjeg ažuriranja i
        // veličine klastera (provjera da nema praznih).
        result.objective.push_back(assign_all(data, result.centroids, params.dist, labels, dists));
        result.final_sizes.assign(nlist, 0);
        for (std::size_t i = 0; i < n; ++i)
            ++result.final_sizes[static_cast<std::size_t>(labels[i])];

        return result;
    }
}
