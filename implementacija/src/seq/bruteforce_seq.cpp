#include "bruteforce.hpp"

#include <stdexcept>

namespace ann
{
    SearchResult bruteforce_knn(const FloatMatrix& base, const FloatMatrix& queries,
                                std::size_t k, DistFn dist)
    {
        if (base.d != queries.d)
            throw std::invalid_argument("bruteforce_knn: baza i upiti imaju razlicite dimenzije");

        SearchResult result(queries.n, k);
        TopK heap(k); // jedan heap, ponovo korišten za svaki upit (bez alokacija)

        for (std::size_t q = 0; q < queries.n; ++q)
        {
            const float* query = queries.row(q);
            for (std::size_t i = 0; i < base.n; ++i)
                heap.push(dist(query, base.row(i), base.d), static_cast<std::int32_t>(i));

            heap.write_sorted(result.ids.data() + q * k, result.distances.data() + q * k);
        }
        return result;
    }
}
