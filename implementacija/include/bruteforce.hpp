#pragma once

// Tačna (brute-force) pretraga k najbližih susjeda: svaki upit se poredi sa
// svakim vektorom baze. O(nq · N · d). Služi kao:
//  - provjera čitača i funkcije udaljenosti (mora se poklopiti sa groundtruth),
//  - referenca "100% recall" sa kojom se poredi IVF.

#include "dataset.hpp"
#include "distance.hpp"
#include "topk.hpp"

#include <cstddef>

namespace ann
{
    SearchResult bruteforce_knn(const FloatMatrix& base, const FloatMatrix& queries,
                                std::size_t k, DistFn dist);
}
