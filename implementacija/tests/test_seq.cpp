// Mali testovi za slučajeve koje pravi podaci (SIFT) ne pokrivaju.
// Pokretanje: build/Release/ann_tests  (ili `ctest --test-dir build/Release`)
//
// Bez test-frameworka, da nema dodatnih zavisnosti: svaki test vraća true/false.

#include "bruteforce.hpp"
#include "eval.hpp"
#include "ivf_index.hpp"
#include "kmeans.hpp"
#include "topk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{
    int failures = 0;

    void expect(bool condition, const char* what)
    {
        std::printf("  [%s] %s\n", condition ? " OK " : "FAIL", what);
        if (!condition)
            ++failures;
    }

    // K-means sa duplikatima: 90 identičnih tačaka (nule) + 20 različitih.
    // Inicijalizacija će skoro sigurno izabrati više nula -> identični
    // centroidi -> svi osim prvog (manji indeks pobjeđuje pri jednakoj
    // udaljenosti) ostaju PRAZNI. Kod za prazne klastere mora ih premjestiti
    // na najdalje tačke, tako da na kraju nema praznih (15 klastera, a 21
    // različita vektora, pa je to moguće).
    void test_kmeans_empty_clusters()
    {
        std::puts("k-means: prazni klasteri (duplikati)");
        constexpr std::size_t n = 110, d = 8, nlist = 15;
        ann::FloatMatrix data;
        data.n = n;
        data.d = d;
        data.values = simd::AlignedFloats(n * d);
        std::memset(data.values.data(), 0, n * d * sizeof(float));
        for (std::size_t i = 90; i < n; ++i)
            for (std::size_t j = 0; j < d; ++j)
                data.values.data()[i * d + j] = static_cast<float>((i - 89) * 10 + j);

        ann::KMeansParams params;
        params.nlist = nlist;
        params.iterations = 10;
        const ann::KMeansResult km = ann::kmeans_train_seq(data, params);

        std::size_t relocated = 0;
        for (const std::size_t r : km.relocated)
            relocated += r;
        std::printf("  (premjesteno praznih klastera ukupno: %zu)\n", relocated);

        expect(relocated > 0, "test zaista pravi prazne klastere");
        expect(std::count(km.final_sizes.begin(), km.final_sizes.end(), std::size_t{ 0 }) == 0,
               "na kraju nema praznih klastera");

        bool finite = true;
        for (std::size_t i = 0; i < nlist * d; ++i)
            finite = finite && std::isfinite(km.centroids.values.data()[i]);
        expect(finite, "centroidi nemaju NaN/inf (nema dijeljenja nulom)");

        bool monotone = true;
        for (std::size_t i = 1; i < km.objective.size(); ++i)
            monotone = monotone && km.objective[i] <= km.objective[i - 1];
        expect(monotone, "ciljna funkcija ne raste");
    }

    void test_kmeans_init()
    {
        std::puts("k-means: inicijalizacija");
        const auto a = ann::kmeans_init_indices(1000, 100, 42);
        const auto b = ann::kmeans_init_indices(1000, 100, 42);
        const auto c = ann::kmeans_init_indices(1000, 100, 43);
        auto sorted = a;
        std::sort(sorted.begin(), sorted.end());
        expect(a == b, "isti seed -> isti indeksi");
        expect(a != c, "drugi seed -> drugi indeksi");
        expect(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "indeksi su razliciti");
        expect(sorted.back() < 1000, "indeksi su u opsegu");
    }

    ann::FloatMatrix random_matrix(std::size_t n, std::size_t d, std::uint64_t seed)
    {
        ann::FloatMatrix m;
        m.n = n;
        m.d = d;
        m.values = simd::AlignedFloats(n * d);
        for (std::size_t i = 0; i < n * d; ++i)
        {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            m.values.data()[i] = static_cast<float>((seed >> 40) % 256); // cijeli 0..255, kao SIFT
        }
        return m;
    }

    // Provjera listi mora PROĆI za ispravan indeks i PASTI kad se struktura
    // namjerno pokvari (inače provjera ništa ne dokazuje).
    void test_ivf_lists_check()
    {
        std::puts("IVF: provjera listi hvata greske");
        const ann::FloatMatrix base = random_matrix(500, 16, 7);
        ann::KMeansParams km;
        km.iterations = 5;
        ann::IvfFlatSeq index(ann::l2_scalar, km);
        index.train(base, 8);
        index.add(base);
        expect(ann::check_ivf_lists(index.lists(), base, ann::l2_scalar).passed(), "ispravan indeks prolazi");

        // Kopija listi koja se može kvariti.
        auto copy_lists = [&]()
        {
            ann::IvfLists l;
            const ann::IvfLists& s = index.lists();
            l.centroids.n = s.centroids.n;
            l.centroids.d = s.centroids.d;
            l.centroids.values = simd::AlignedFloats(s.centroids.n * s.centroids.d);
            std::memcpy(l.centroids.values.data(), s.centroids.values.data(), s.centroids.n * s.centroids.d * sizeof(float));
            l.offset = s.offset;
            l.list_data = simd::AlignedFloats(s.n * s.d());
            std::memcpy(l.list_data.data(), s.list_data.data(), s.n * s.d() * sizeof(float));
            l.list_ids = s.list_ids;
            l.n = s.n;
            return l;
        };

        {
            ann::IvfLists l = copy_lists();
            l.offset.back() -= 1;
            expect(!ann::check_ivf_lists(l, base, ann::l2_scalar).offset_ok, "pogresan offset[nlist] se otkriva");
        }
        {
            ann::IvfLists l = copy_lists();
            l.list_ids[3] = l.list_ids[4];
            expect(!ann::check_ivf_lists(l, base, ann::l2_scalar).ids_permutation, "duplikat ID-a se otkriva");
        }
        {
            ann::IvfLists l = copy_lists();
            l.list_data.data()[5] += 1.0f;
            expect(!ann::check_ivf_lists(l, base, ann::l2_scalar).data_ok, "izmijenjen vektor se otkriva");
        }
        {
            // Zamjena prvog vektora iz liste 0 i prvog iz liste 1 (podaci i ID
            // zajedno): struktura je konzistentna, ali vektori su u pogrešnim listama.
            ann::IvfLists l = copy_lists();
            const std::size_t d = l.d();
            const std::size_t a = l.offset[0], b = l.offset[1];
            for (std::size_t j = 0; j < d; ++j)
                std::swap(l.list_data.data()[a * d + j], l.list_data.data()[b * d + j]);
            std::swap(l.list_ids[a], l.list_ids[b]);
            expect(!ann::check_ivf_lists(l, base, ann::l2_scalar).nearest_ok, "vektor u pogresnoj listi se otkriva");
        }
    }

    // Pretraga: nprobe == nlist mora biti identična brute-force-u (podaci su
    // cijeli brojevi 0..255, pa ima mnogo jednakih udaljenosti i tie-break se
    // zaista testira). nprobe = 1 skenira tačno jednu listu.
    void test_ivf_search()
    {
        std::puts("IVF: pretraga");
        const ann::FloatMatrix base = random_matrix(600, 4, 11); // d = 4: mnogo jednakih udaljenosti
        const ann::FloatMatrix queries = random_matrix(50, 4, 12);
        ann::KMeansParams km;
        km.iterations = 5;
        ann::IvfFlatSeq index(ann::l2_scalar, km);
        index.train(base, 10);
        index.add(base);

        const std::size_t k = 20;
        std::size_t scanned = 0;
        const ann::SearchResult full = index.search(queries, k, 10, &scanned);
        const ann::SearchResult bf = ann::bruteforce_knn(base, queries, k, ann::l2_scalar);
        expect(full.ids == bf.ids && full.distances == bf.distances, "nprobe == nlist: identicno brute-force-u");
        expect(scanned == queries.n * base.n, "nprobe == nlist: skenirana cijela baza za svaki upit");

        const ann::SearchResult over = index.search(queries, k, 1000, nullptr);
        expect(over.ids == full.ids, "nprobe > nlist se svodi na nlist");

        bool ties = false;
        for (std::size_t i = 1; i < bf.distances.size(); ++i)
            ties = ties || (i % k != 0 && bf.distances[i] == bf.distances[i - 1]);
        expect(ties, "test zaista sadrzi jednake udaljenosti");

        // nprobe = 1: broj skeniranih = zbir dužina listi najbližeg centroida.
        const ann::IvfLists& L = index.lists();
        std::size_t expected = 0;
        std::vector<ann::Neighbor> probes;
        for (std::size_t q = 0; q < queries.n; ++q)
        {
            ann::select_probes(queries.row(q), L.centroids, ann::l2_scalar, 1, probes);
            expected += L.list_size(static_cast<std::size_t>(probes[0].id));
        }
        index.search(queries, k, 1, &scanned);
        expect(scanned == expected, "nprobe = 1: skenira se samo najbliza lista");

        // Više kandidata nego u listi: k = 600 > dužina bilo koje liste.
        const ann::SearchResult padded = index.search(queries, 600, 1, nullptr);
        bool pad_ok = true;
        for (std::size_t q = 0; q < queries.n; ++q)
        {
            const std::int32_t* ids = padded.ids.data() + q * 600;
            pad_ok = pad_ok && ids[0] >= 0 && ids[599] == -1;
        }
        expect(pad_ok, "manje kandidata od k: ostatak je -1");
    }

    void test_topk_ties()
    {
        std::puts("top-k: jednake udaljenosti i manje od k kandidata");
        ann::TopK heap(3);
        // Ubacujemo obrnutim redom ID-eva: pri jednakoj udaljenosti mora
        // pobijediti manji ID, bez obzira na redoslijed ubacivanja.
        for (const int id : { 9, 7, 5, 3, 1 })
            heap.push(4.0f, id);
        heap.push(1.0f, 8);
        std::int32_t ids[3];
        float dists[3];
        heap.write_sorted(ids, dists);
        expect(ids[0] == 8 && ids[1] == 1 && ids[2] == 3, "redoslijed 8, 1, 3");
        expect(dists[0] == 1.0f && dists[1] == 4.0f && dists[2] == 4.0f, "udaljenosti rastu");

        heap.push(2.0f, 5);
        heap.write_sorted(ids, dists);
        expect(ids[0] == 5 && ids[1] == -1 && ids[2] == -1, "nedostajuci su -1");
        expect(std::isinf(dists[1]) && std::isinf(dists[2]), "nedostajuci imaju +inf");
    }
}

int main()
{
    test_kmeans_init();
    test_kmeans_empty_clusters();
    test_topk_ties();
    test_ivf_lists_check();
    test_ivf_search();
    std::printf("\n%s (%d gresaka)\n", failures == 0 ? "SVI TESTOVI PROSLI" : "NEKI TESTOVI NISU PROSLI", failures);
    return failures == 0 ? 0 : 1;
}
