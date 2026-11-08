#pragma once

// Mjere kvaliteta rezultata pretrage.

#include "dataset.hpp"
#include "distance.hpp"
#include "topk.hpp"

#include <cstddef>
#include <cstdint>

namespace ann
{
    // Recall@k = prosjek po upitima od |naših k ∩ tačnih k| / k.
    // Koristi se prvih k kolona i iz rezultata i iz groundtruth-a.
    double recall_at_k(const SearchResult& result, const IntMatrix& groundtruth, std::size_t k);

    // Provjera da je rezultat TAČAN kNN, upoređivanjem sa groundtruth-om.
    //
    // Poređenje samo ID-eva nije dovoljno: ako dva vektora imaju istu udaljenost
    // od upita, i naš i groundtruth rezultat su tačni, a redoslijed (ili čak
    // koji od njih uđe u top-k na granici) može biti različit. Zato se za
    // svaku poziciju j poredi udaljenost našeg j-tog susjeda sa udaljenošću
    // groundtruth j-tog susjeda (izračunatom istom funkcijom). Ako se sve
    // udaljenosti poklapaju, rezultat je tačan.
    struct GroundtruthCheck
    {
        std::size_t queries = 0;            // broj provjerenih upita
        std::size_t k = 0;                  // dubina provjere
        std::size_t distances_match = 0;    // upiti kod kojih se sve udaljenosti poklapaju
        std::size_t ids_match = 0;          // upiti kod kojih se poklapaju i ID-evi, istim redom
        // Prvo neslaganje (za ispis), ako postoji:
        std::int64_t first_bad_query = -1;
        std::size_t first_bad_pos = 0;
        float ours = 0, expected = 0;

        bool passed() const
        {
            return distances_match == queries;
        }
    };

    // Provjera invertovanih listi (korak 3).
    struct IvfListsCheck
    {
        bool offset_ok = false;        // offset[0] == 0, neopadajući, offset[nlist] == N
        bool sizes_sum_ok = false;     // zbir dužina listi == N
        bool ids_permutation = false;  // svaki ID 0..N-1 se javlja tačno jednom
        bool ids_sorted = false;       // ID-evi rastu unutar svake liste (deterministički raspored)
        bool data_ok = false;          // list_data na poziciji p == base[list_ids[p]]
        bool nearest_ok = false;       // svaki provjereni vektor je u listi svog najbližeg centroida
        std::size_t nearest_checked = 0;

        // Statistika listi:
        std::size_t min_size = 0, max_size = 0, empty_lists = 0;
        double imbalance = 0;          // nlist · Σ|L_i|² / N²

        bool passed() const
        {
            return offset_ok && sizes_sum_ok && ids_permutation && ids_sorted && data_ok && nearest_ok;
        }
    };

    // nearest_stride: provjerava se najbliži centroid za svaki nearest_stride-ti
    // vektor (1 = svi). Ta provjera košta kao još jedno punjenje, pa se za
    // SIFT1M uzima uzorak.
    struct IvfLists;
    IvfListsCheck check_ivf_lists(const IvfLists& lists, const FloatMatrix& base,
                                  DistFn dist, std::size_t nearest_stride = 1);

    GroundtruthCheck check_against_groundtruth(const SearchResult& result,
                                               const IntMatrix& groundtruth,
                                               const FloatMatrix& base,
                                               const FloatMatrix& queries,
                                               DistFn dist);
}
