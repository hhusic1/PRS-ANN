#include "eval.hpp"

#include "ivf_index.hpp"
#include "kmeans.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ann
{
    IvfListsCheck check_ivf_lists(const IvfLists& lists, const FloatMatrix& base,
                                  DistFn dist, std::size_t nearest_stride)
    {
        IvfListsCheck check;
        const std::size_t nlist = lists.nlist();
        const std::size_t n = base.n;
        const std::size_t d = base.d;
        if (nearest_stride == 0)
            nearest_stride = 1;

        // offset: dužina nlist + 1, počinje od 0, ne opada, završava na N.
        check.offset_ok = lists.offset.size() == nlist + 1 && lists.offset.front() == 0 &&
                          lists.offset.back() == n && lists.n == n &&
                          std::is_sorted(lists.offset.begin(), lists.offset.end());
        if (!check.offset_ok)
            return check; // ostale provjere bez ispravnog offseta nemaju smisla

        std::size_t sum = 0;
        double sum_sq = 0.0;
        check.min_size = n;
        for (std::size_t c = 0; c < nlist; ++c)
        {
            const std::size_t s = lists.list_size(c);
            sum += s;
            sum_sq += static_cast<double>(s) * static_cast<double>(s);
            check.min_size = std::min(check.min_size, s);
            check.max_size = std::max(check.max_size, s);
            check.empty_lists += s == 0 ? 1 : 0;
        }
        check.sizes_sum_ok = sum == n;
        check.imbalance = static_cast<double>(nlist) * sum_sq / (static_cast<double>(n) * static_cast<double>(n));

        // Svaki ID tačno jednom.
        std::vector<unsigned char> seen(n, 0);
        check.ids_permutation = lists.list_ids.size() == n;
        for (std::size_t p = 0; p < lists.list_ids.size() && check.ids_permutation; ++p)
        {
            const std::int32_t id = lists.list_ids[p];
            if (id < 0 || static_cast<std::size_t>(id) >= n || seen[static_cast<std::size_t>(id)])
                check.ids_permutation = false;
            else
                seen[static_cast<std::size_t>(id)] = 1;
        }

        // ID-evi rastu unutar liste; podaci odgovaraju bazi.
        check.ids_sorted = true;
        check.data_ok = check.ids_permutation;
        for (std::size_t c = 0; c < nlist; ++c)
        {
            const std::int32_t* ids = lists.list_vector_ids(c);
            const std::size_t s = lists.list_size(c);
            check.ids_sorted = check.ids_sorted && std::is_sorted(ids, ids + s);
            for (std::size_t j = 0; j < s && check.data_ok; ++j)
            {
                const float* stored = lists.list_vectors(c) + j * d;
                if (std::memcmp(stored, base.row(static_cast<std::size_t>(ids[j])), d * sizeof(float)) != 0)
                    check.data_ok = false;
            }
        }

        // Vektor u listi c mora imati c kao najbliži centroid (uzorak po poziciji).
        check.nearest_ok = true;
        for (std::size_t c = 0; c < nlist; ++c)
        {
            for (std::size_t p = lists.offset[c]; p < lists.offset[c + 1]; ++p)
            {
                if (p % nearest_stride != 0)
                    continue;
                ++check.nearest_checked;
                const float* x = lists.list_data.data() + p * d;
                if (nearest_centroid(x, lists.centroids, dist) != c)
                    check.nearest_ok = false;
            }
        }
        return check;
    }

    double recall_at_k(const SearchResult& result, const IntMatrix& groundtruth, std::size_t k)
    {
        if (k == 0 || k > result.k || k > groundtruth.d)
            throw std::invalid_argument("recall_at_k: k je veci od broja dostupnih susjeda");
        if (groundtruth.n < result.nq)
            throw std::invalid_argument("recall_at_k: groundtruth ima manje upita od rezultata");

        std::size_t hits = 0;
        std::vector<std::int32_t> truth(k);
        for (std::size_t q = 0; q < result.nq; ++q)
        {
            // k je malo (10, 100), pa sortiranje + binarna pretraga je dovoljno.
            std::copy_n(groundtruth.row(q), k, truth.begin());
            std::sort(truth.begin(), truth.end());

            const std::int32_t* ours = result.ids.data() + q * result.k;
            for (std::size_t j = 0; j < k; ++j)
                hits += std::binary_search(truth.begin(), truth.end(), ours[j]) ? 1 : 0;
        }
        return static_cast<double>(hits) / static_cast<double>(result.nq * k);
    }

    GroundtruthCheck check_against_groundtruth(const SearchResult& result,
                                               const IntMatrix& groundtruth,
                                               const FloatMatrix& base,
                                               const FloatMatrix& queries,
                                               DistFn dist)
    {
        GroundtruthCheck check;
        check.queries = result.nq;
        check.k = std::min(result.k, groundtruth.d);

        if (groundtruth.n < result.nq || queries.n < result.nq)
            throw std::invalid_argument("check_against_groundtruth: premalo upita u groundtruth-u");

        for (std::size_t q = 0; q < result.nq; ++q)
        {
            const std::int32_t* ours_ids = result.ids.data() + q * result.k;
            const float* ours_dist = result.distances.data() + q * result.k;
            const std::int32_t* gt_ids = groundtruth.row(q);

            bool dist_ok = true;
            bool ids_ok = true;
            for (std::size_t j = 0; j < check.k; ++j)
            {
                const auto gt_id = static_cast<std::size_t>(gt_ids[j]);
                if (gt_id >= base.n)
                    throw std::runtime_error("groundtruth sadrzi ID izvan baze");

                const float expected = dist(queries.row(q), base.row(gt_id), base.d);
                // Poređenje na tačnu jednakost je namjerno: za SIFT podatke
                // udaljenost je tačna (vidi l2_scalar.cpp), pa svako odstupanje
                // znači grešku, a ne zaokruživanje.
                if (ours_dist[j] != expected)
                {
                    if (dist_ok && check.first_bad_query < 0)
                    {
                        check.first_bad_query = static_cast<std::int64_t>(q);
                        check.first_bad_pos = j;
                        check.ours = ours_dist[j];
                        check.expected = expected;
                    }
                    dist_ok = false;
                }
                if (ours_ids[j] != gt_ids[j])
                    ids_ok = false;
            }
            check.distances_match += dist_ok ? 1 : 0;
            check.ids_match += ids_ok ? 1 : 0;
        }
        return check;
    }
}
