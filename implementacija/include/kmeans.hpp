#pragma once

// K-means (Lloydov algoritam) za treniranje IVF centroida (grubi kvantizator).
//
// Svaka iteracija ima dva koraka:
//  1. dodjela:    svaka tačka ide najbližem centroidu,
//  2. ažuriranje: svaki centroid postaje prosjek svojih tačaka.
// Ciljna funkcija (zbir kvadrata udaljenosti tačaka do njihovih centroida)
// nikad ne raste: dodjela je minimizuje za fiksne centroide, a ažuriranje za
// fiksnu dodjelu. To je provjera ispravnosti za korak 2.

#include "dataset.hpp"
#include "distance.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ann
{
    struct KMeansParams
    {
        std::size_t nlist = 0;              // broj klastera (invertovanih listi)
        int iterations = 25;                // fiksan broj iteracija (bez kriterija konvergencije,
                                            // da sve verzije rade isti posao i da se mogu porediti)
        std::uint64_t seed = 20261107;      // fiksan seed -> isti centroidi pri svakom pokretanju
        DistFn dist = l2_scalar;
    };

    struct KMeansResult
    {
        FloatMatrix centroids;                // nlist × d

        // objective[i] = zbir kvadrata udaljenosti za dodjelu u iteraciji i
        // (prije ažuriranja); zadnji element je za konačnu dodjelu nakon
        // posljednjeg ažuriranja. Ukupno iterations + 1 vrijednosti.
        std::vector<double> objective;

        std::vector<std::size_t> relocated;   // broj praznih klastera popravljenih u svakoj iteraciji
        std::vector<std::size_t> final_sizes; // veličina svakog klastera u konačnoj dodjeli
    };

    // Indeksi nlist različitih tačaka iz [0, n), izabranih nasumično sa
    // fiksnim seedom. Odvojeno od treninga da i OpenMP verzija koristi
    // potpuno istu inicijalizaciju.
    std::vector<std::size_t> kmeans_init_indices(std::size_t n, std::size_t nlist, std::uint64_t seed);

    // Indeks najbližeg centroida (pri jednakoj udaljenosti manji indeks).
    // U best_dist (ako nije nullptr) upisuje udaljenost do njega.
    // Koristi se i u k-meansu i pri punjenju invertovanih listi.
    std::size_t nearest_centroid(const float* x, const FloatMatrix& centroids,
                                 DistFn dist, float* best_dist = nullptr);

    KMeansResult kmeans_train_seq(const FloatMatrix& data, const KMeansParams& params);
}
