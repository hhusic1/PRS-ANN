#pragma once

// Top-k selekcija i zajednički format rezultata pretrage.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ann
{
    struct Neighbor
    {
        float dist;
        std::int32_t id;
    };

    // Strogi potpuni poredak kandidata: manja udaljenost je bolja, a pri
    // JEDNAKOJ udaljenosti bolji je manji ID. Bez tog pravila redoslijed
    // susjeda sa istom udaljenošću zavisi od redoslijeda obilaska i od
    // standardne biblioteke (libstdc++ / libc++), pa se rezultati sekvencijalne,
    // SIMD i OpenMP verzije (i Windows / macOS) ne bi mogli porediti ID po ID.
    inline bool better(const Neighbor& a, const Neighbor& b)
    {
        return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
    }

    // Max-heap veličine k: vrh (front) je NAJLOŠIJI od k trenutno najboljih.
    // Novi kandidat se poredi samo sa vrhom: ako nije bolji, odbacuje se u O(1);
    // ako jeste, zamjenjuje vrh u O(log k). Ukupno O(N log k) umjesto O(N log N)
    // za puno sortiranje, a memorija je O(k).
    //
    // std::push_heap / pop_heap sa komparatorom `better` (koji se ponaša kao
    // "manje od") drže na vrhu "najveći" element, tj. najlošijeg kandidata.
    class TopK
    {
    public:
        explicit TopK(std::size_t k) : _k(k)
        {
            _heap.reserve(k);
        }

        void clear()
        {
            _heap.clear();
        }

        void push(float dist, std::int32_t id)
        {
            const Neighbor candidate{ dist, id };
            if (_heap.size() < _k)
            {
                _heap.push_back(candidate);
                std::push_heap(_heap.begin(), _heap.end(), better);
            }
            else if (_k > 0 && better(candidate, _heap.front()))
            {
                std::pop_heap(_heap.begin(), _heap.end(), better); // najlošiji ide na kraj
                _heap.back() = candidate;
                std::push_heap(_heap.begin(), _heap.end(), better);
            }
        }

        // Sortira rastuće (najbolji prvi) i upisuje k rezultata. Ako je bilo
        // manje od k kandidata (npr. premalo vektora u ispitanim listama),
        // ostatak se popunjava sa id = -1 i dist = +inf, kao u FAISS-u.
        // Nakon poziva heap je prazan i objekat se može ponovo koristiti.
        void write_sorted(std::int32_t* ids, float* dists)
        {
            std::sort_heap(_heap.begin(), _heap.end(), better);
            for (std::size_t j = 0; j < _k; ++j)
            {
                if (j < _heap.size())
                {
                    ids[j] = _heap[j].id;
                    dists[j] = _heap[j].dist;
                }
                else
                {
                    ids[j] = -1;
                    dists[j] = std::numeric_limits<float>::infinity();
                }
            }
            _heap.clear();
        }

    private:
        std::size_t _k;
        std::vector<Neighbor> _heap;
    };

    // Rezultat pretrage za nq upita: red q (k elemenata) su susjedi upita q,
    // sortirani rastuće po udaljenosti.
    struct SearchResult
    {
        std::size_t nq = 0;
        std::size_t k = 0;
        std::vector<std::int32_t> ids;  // nq × k
        std::vector<float> distances;   // nq × k (kvadrat L2)

        SearchResult() = default;
        SearchResult(std::size_t nq_, std::size_t k_)
            : nq(nq_), k(k_), ids(nq_ * k_), distances(nq_ * k_)
        {
        }
    };
}
