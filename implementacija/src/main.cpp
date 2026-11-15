// PRS-ANN: ulazna tačka programa.
//
// Komande (po koracima iz plana):
//
//   prs_ann gt-check <folder_skupa> [broj_upita] [k]          korak 1
//   prs_ann kmeans-check <folder_skupa> <nlist> [iteracije] [seed]   korak 2
//   prs_ann ivf-check <folder_skupa> <nlist> [iteracije] [seed]      korak 3
//   prs_ann ivf-search <folder_skupa> <nlist> [broj_upita] [k]       korak 4
//   prs_ann ivf-bench <folder_skupa> <nlist> <nprobe> [broj_upita] [ponavljanja] [udaljenost] [provjera]
//                                                     mjerenje pretrage (docs/vtune/)
//
// Primjer (pokrenuti iz foldera implementacija/):
//   build/Release/prs_ann gt-check data/siftsmall
//   build/Release/prs_ann gt-check data/sift 100
//   build/Release/prs_ann kmeans-check data/siftsmall 64
//
// Ime skupa se uzima iz imena foldera: data/siftsmall -> siftsmall_base.fvecs,
// siftsmall_query.fvecs, siftsmall_groundtruth.ivecs. Putanja je argument, a
// ne konstanta u kodu, da isti program radi na Windowsu i na macOS-u.

#include "bench.hpp"
#include "bruteforce.hpp"
#include "dataset.hpp"
#include "distance.hpp"
#include "eval.hpp"
#include "ivf_index.hpp"
#include "ivf_io.hpp"
#include "kmeans.hpp"

#include <algorithm>
#include <numeric>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace fs = std::filesystem;

    void print_usage()
    {
        std::puts("Upotreba:\n"
                  "  prs_ann gt-check <folder_skupa> [broj_upita] [k]\n"
                  "  prs_ann kmeans-check <folder_skupa> <nlist> [iteracije] [seed]\n"
                  "  prs_ann ivf-check <folder_skupa> <nlist> [iteracije] [seed]\n"
                  "  prs_ann ivf-search <folder_skupa> <nlist> [broj_upita] [k]   (k podrazumijevano 10)\n"
                  "  prs_ann ivf-bench <folder_skupa> <nlist> <nprobe> [broj_upita] [ponavljanja] [udaljenost] [provjera]\n"
                  "                (podrazumijevano 1000 upita, 5 ponavljanja, udaljenost scalar, provjera 1;\n"
                  "                 provjera 0 preskace poredjenje sa skalarnom referencom, za VTune)\n"
                  "\n"
                  "  folder_skupa  npr. data/siftsmall ili data/sift\n"
                  "  broj_upita    koliko prvih upita provjeriti (0 = svi; podrazumijevano 0)\n"
                  "  k             broj susjeda (podrazumijevano = dubina groundtruth-a, 100)\n"
                  "  nlist         broj klastera (npr. 16, 64, 128 za SIFT10K; 256, 1024, 4096 za SIFT1M)\n"
                  "  iteracije     broj k-means iteracija (podrazumijevano 25)\n"
                  "  seed          seed za inicijalizaciju (podrazumijevano 20261107)");
    }

    double seconds_since(std::chrono::steady_clock::time_point start)
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    const char* build_name()
    {
#if defined(NDEBUG)
        return "Release";
#else
        return "Debug - vrijeme nije za mjerenje";
#endif
    }

    // Korak 3: trening (learn) + punjenje invertovanih listi (base) i provjera
    // CSR strukture.
    int run_ivf_check(const fs::path& dir, std::size_t nlist, int iterations, std::uint64_t seed)
    {
        const std::string name = fs::path(dir).filename().string();
        const ann::FloatMatrix learn = ann::read_fvecs(dir / (name + "_learn.fvecs"));
        const ann::FloatMatrix base = ann::read_fvecs(dir / (name + "_base.fvecs"));
        std::printf("Skup:      %s, learn = %zu, base = %zu, d = %zu\n", name.c_str(), learn.n, base.n, base.d);
        std::printf("IVF:       nlist = %zu, k-means %d iteracija, seed %llu (%s)\n\n", nlist, iterations,
                    static_cast<unsigned long long>(seed), build_name());

        ann::KMeansParams km;
        km.iterations = iterations;
        km.seed = seed;
        ann::IvfFlatSeq index(ann::l2_scalar, km);

        auto t0 = std::chrono::steady_clock::now();
        const ann::KMeansResult stats = index.train(learn, nlist);
        const double train_s = seconds_since(t0);

        t0 = std::chrono::steady_clock::now();
        index.add(base);
        const double add_s = seconds_since(t0);

        std::printf("train:     %.2f s (ciljna funkcija %.1f -> %.1f)\n", train_s, stats.objective.front(),
                    stats.objective.back());
        std::printf("add:       %.2f s (%.0f vektora/s)\n", add_s, static_cast<double>(base.n) / add_s);

        // Za velike baze provjera najbližeg centroida ide na uzorku (~10.000).
        const std::size_t stride = base.n > 100'000 ? base.n / 10'000 : 1;
        const ann::IvfListsCheck check = ann::check_ivf_lists(index.lists(), base, ann::l2_scalar, stride);
        const auto& L = index.lists();

        auto mark = [](bool ok) { return ok ? "OK" : "GRESKA"; };
        std::printf("\nProvjera CSR strukture:\n");
        std::printf("  offset[0] = %zu, offset[nlist] = %zu, N = %zu, neopadajuci   %s\n",
                    L.offset.empty() ? 0 : L.offset.front(), L.offset.empty() ? 0 : L.offset.back(), base.n,
                    mark(check.offset_ok));
        std::printf("  zbir duzina listi == N                                  %s\n", mark(check.sizes_sum_ok));
        std::printf("  svaki ID tacno jednom                                   %s\n", mark(check.ids_permutation));
        std::printf("  ID-evi rastu unutar liste                               %s\n", mark(check.ids_sorted));
        std::printf("  list_data[p] == base[list_ids[p]]                       %s\n", mark(check.data_ok));
        std::printf("  vektor je u listi najblizeg centroida (%zu provjereno)  %s\n", check.nearest_checked,
                    mark(check.nearest_ok));

        std::printf("\nListe (base): min %zu, max %zu, prazne %zu, prosjek %.1f\n", check.min_size, check.max_size,
                    check.empty_lists, static_cast<double>(base.n) / static_cast<double>(nlist));
        std::printf("Imbalance faktor (base):  %.3f\n", check.imbalance);
        std::printf("Memorija listi:           %.1f MB\n",
                    static_cast<double>(base.n * (base.d * sizeof(float) + sizeof(std::int32_t))) / 1e6);

        std::puts(check.passed() ? "\nPROSLO: invertovane liste su ispravne." : "\nNIJE PROSLO.");
        return check.passed() ? 0 : 1;
    }

    // Korak 4: pretraga. Za svaki nprobe ispisuje Recall@10, broj upita sa
    // tačnim top-k (po udaljenostima) i orijentaciono vrijeme. Uslov prolaza:
    // sa nprobe == nlist pretraga je tačna, tj.
    //   (a) udaljenosti na svim pozicijama se poklapaju sa groundtruth-om, i
    //   (b) rezultat (ID-evi i udaljenosti) je identičan brute-force-u.
    // (b) je jače od "recall = 1.0": kod jednakih udaljenosti groundtruth
    // može imati drugi ID od nas (vidi korak 1), a brute-force koristi isto
    // pravilo (manji ID), pa mora biti poklapanje ID po ID.
    int run_ivf_search(const fs::path& dir, std::size_t nlist, std::size_t max_queries, std::size_t k)
    {
        const std::string name = fs::path(dir).filename().string();
        const ann::FloatMatrix learn = ann::read_fvecs(dir / (name + "_learn.fvecs"));
        const ann::FloatMatrix base = ann::read_fvecs(dir / (name + "_base.fvecs"));
        const ann::FloatMatrix queries = ann::read_fvecs(dir / (name + "_query.fvecs"), max_queries);
        const ann::IntMatrix gt = ann::read_ivecs(dir / (name + "_groundtruth.ivecs"), max_queries);
        if (k == 0)
            k = 10;
        if (k > gt.d)
            throw std::invalid_argument("k je veci od dubine groundtruth-a");

        std::printf("Skup:      %s, learn = %zu, base = %zu, upita = %zu, d = %zu\n", name.c_str(), learn.n,
                    base.n, queries.n, base.d);
        std::printf("IVF:       nlist = %zu, k = %zu, k-means 25 iteracija (%s)\n\n", nlist, k, build_name());

        ann::IvfFlatSeq index;
        auto t0 = std::chrono::steady_clock::now();
        index.train(learn, nlist);
        const double train_s = seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        index.add(base);
        const double add_s = seconds_since(t0);
        std::printf("train %.2f s, add %.2f s\n\n", train_s, add_s);

        std::vector<std::size_t> nprobes;
        for (const std::size_t p : { 1, 8, 32, 128 })
            if (p < nlist)
                nprobes.push_back(p);
        nprobes.push_back(nlist);

        // Vrijeme je ovdje samo orijentaciono (jedno pokretanje, bez
        // zagrijavanja); prava mjerenja su u fazi mjerenja (perf::measure).
        std::puts(" nprobe   Recall@10   tacan top-k   skenirano/upit (% baze)    vrijeme      QPS");
        ann::SearchResult full;
        ann::GroundtruthCheck full_check;
        for (const std::size_t nprobe : nprobes)
        {
            std::size_t scanned = 0;
            t0 = std::chrono::steady_clock::now();
            ann::SearchResult result = index.search(queries, k, nprobe, &scanned);
            const double search_s = seconds_since(t0);

            const ann::GroundtruthCheck check =
                ann::check_against_groundtruth(result, gt, base, queries, ann::l2_scalar);
            const double per_query = static_cast<double>(scanned) / static_cast<double>(queries.n);
            std::printf(" %6zu   %9.4f   %5zu/%-5zu   %10.0f (%6.2f %%)    %7.3f s  %8.1f\n", nprobe,
                        ann::recall_at_k(result, gt, std::min<std::size_t>(10, k)), check.distances_match,
                        check.queries, per_query, 100.0 * per_query / static_cast<double>(base.n), search_s,
                        static_cast<double>(queries.n) / search_s);
            if (nprobe == nlist)
            {
                full = std::move(result);
                full_check = check;
            }
        }

        // Brute-force za poređenje ID po ID. Na SIFT1M je skup (~0,2 s po
        // upitu), pa se poredi prvih 100 upita.
        const std::size_t bf_n = base.n > 100'000 ? std::min<std::size_t>(queries.n, 100) : queries.n;
        const ann::FloatMatrix bf_queries = ann::read_fvecs(dir / (name + "_query.fvecs"), bf_n);
        const ann::SearchResult bf = ann::bruteforce_knn(base, bf_queries, k, ann::l2_scalar);
        std::size_t identical = 0;
        for (std::size_t q = 0; q < bf_n; ++q)
        {
            const std::size_t from = q * k;
            identical += std::equal(bf.ids.begin() + from, bf.ids.begin() + from + k, full.ids.begin() + from) &&
                                 std::equal(bf.distances.begin() + from, bf.distances.begin() + from + k,
                                            full.distances.begin() + from)
                             ? 1
                             : 0;
        }

        std::printf("\nProvjera za nprobe == nlist = %zu:\n", nlist);
        std::printf("  udaljenosti == groundtruth (svih %zu pozicija): %zu / %zu upita\n", k,
                    full_check.distances_match, full_check.queries);
        std::printf("  identicno brute-force-u (ID-evi i udaljenosti): %zu / %zu upita\n", identical, bf_n);

        const bool ok = full_check.passed() && identical == bf_n;
        std::puts(ok ? "\nPROSLO: sa nprobe == nlist pretraga je tacna." : "\nNIJE PROSLO.");
        return ok ? 0 : 1;
    }

    // Korak 2: k-means na learn skupu. Ciljna funkcija mora opadati po
    // iteracijama, a na kraju ne smije biti praznih klastera.
    int run_kmeans_check(const fs::path& dir, std::size_t nlist, int iterations, std::uint64_t seed)
    {
        const std::string name = fs::path(dir).filename().string();
        const ann::FloatMatrix learn = ann::read_fvecs(dir / (name + "_learn.fvecs"));
        std::printf("Skup:      %s, learn = %zu vektora, d = %zu\n", name.c_str(), learn.n, learn.d);
        std::printf("K-means:   nlist = %zu, iteracija = %d, seed = %llu (%.1f tacaka po klasteru)\n",
                    nlist, iterations, static_cast<unsigned long long>(seed),
                    static_cast<double>(learn.n) / static_cast<double>(nlist));

        ann::KMeansParams params;
        params.nlist = nlist;
        params.iterations = iterations;
        params.seed = seed;

        const auto t0 = std::chrono::steady_clock::now();
        const ann::KMeansResult km = ann::kmeans_train_seq(learn, params);
        const double train_s = seconds_since(t0);
        std::printf("Trening:   %.2f s (l2_scalar, %s)\n\n", train_s, build_name());

        // Ciljna funkcija po iteracijama. Teorijski ne raste nikad; dozvoljava se
        // samo relativni porast do 1e-6 zbog zaokruživanja u floatu (centroidi
        // nisu cijeli brojevi, pa udaljenosti više nisu tačne kao u koraku 1).
        constexpr double tolerance = 1e-6;
        bool monotone = true;
        double worst_increase = 0.0;
        std::puts(" iter   ciljna funkcija    rel. promjena   prazni");
        for (std::size_t i = 0; i < km.objective.size(); ++i)
        {
            const double value = km.objective[i];
            const bool is_final = i + 1 == km.objective.size();
            if (i == 0)
                std::printf(" %4zu  %16.1f   %13s   %6zu\n", i, value, "-", km.relocated[i]);
            else
            {
                const double prev = km.objective[i - 1];
                const double rel = (value - prev) / prev;
                worst_increase = std::max(worst_increase, rel);
                if (rel > tolerance)
                    monotone = false;
                if (is_final)
                    std::printf(" kraj  %16.1f   %+13.3e\n", value, rel);
                else
                    std::printf(" %4zu  %16.1f   %+13.3e   %6zu\n", i, value, rel, km.relocated[i]);
            }
        }

        const auto& sizes = km.final_sizes;
        const std::size_t empty = static_cast<std::size_t>(std::count(sizes.begin(), sizes.end(), std::size_t{ 0 }));
        const auto [min_it, max_it] = std::minmax_element(sizes.begin(), sizes.end());
        // Imbalance faktor = nlist · Σ|L_i|² / N². 1.0 = sve liste iste dužine.
        // Mjeri koliko je pretraga skuplja od idealne zbog neravnomjernih listi.
        double sum_sq = 0.0;
        for (const std::size_t s : sizes)
            sum_sq += static_cast<double>(s) * static_cast<double>(s);
        const double imbalance = static_cast<double>(nlist) * sum_sq /
                                 (static_cast<double>(learn.n) * static_cast<double>(learn.n));
        const std::size_t total = std::accumulate(sizes.begin(), sizes.end(), std::size_t{ 0 });

        std::printf("\nVelicine klastera (learn): min %zu, max %zu, prazni %zu, zbir %zu\n",
                    *min_it, *max_it, empty, total);
        std::printf("Imbalance faktor:          %.3f\n", imbalance);
        std::printf("Najveci relativni porast:  %.3e (tolerancija %.0e)\n", worst_increase, tolerance);

        const bool ok = monotone && empty == 0 && total == learn.n;
        std::puts(ok ? "\nPROSLO: ciljna funkcija ne raste, nema praznih klastera."
                     : "\nNIJE PROSLO.");
        return ok ? 0 : 1;
    }

    // Mjerenje pretrage za VTune analizu (docs/vtune/). Indeks se jednom
    // napravi SKALARNOM verzijom i sačuva u folder skupa, pa sve varijante
    // udaljenosti i sva ponovljena mjerenja rade na identičnom indeksu (brža
    // udaljenost bi pri izgradnji dala centroide različite u zadnjim bitima).
    // Mjeri se samo pretraga, kroz perf::measure (zagrijavanje, ponavljanja,
    // medijan). Izlazni kod 1 ako se rezultat razlikuje od skalarne reference.
    // check = false preskače skalarnu referencu: pod VTune-om bi njeno
    // računanje ušlo u profil optimizovane varijante.
    int run_ivf_bench(const fs::path& dir, std::size_t nlist, std::size_t nprobe, std::size_t max_queries,
                      int repeats, const std::string& dist_name, bool check)
    {
        const ann::DistanceVariant* variant = ann::find_distance(dist_name);
        if (variant == nullptr)
        {
            std::fprintf(stderr, "Nepoznata udaljenost '%s'. Moguce:", dist_name.c_str());
            for (const ann::DistanceVariant& v : ann::distance_variants())
                std::fprintf(stderr, " %s", v.name);
            std::fputc('\n', stderr);
            return 2;
        }
        if (variant->needs_avx2_fma && !ann::cpu_has_avx2_fma())
        {
            std::fprintf(stderr, "Udaljenost '%s' trazi procesor sa AVX2 i FMA, a ovaj ih nema.\n", variant->name);
            return 2;
        }
        if (repeats < 1)
            throw std::invalid_argument("ponavljanja mora biti >= 1");

        const std::string name = fs::path(dir).filename().string();
        const ann::FloatMatrix base = ann::read_fvecs(dir / (name + "_base.fvecs"));
        const ann::FloatMatrix queries = ann::read_fvecs(dir / (name + "_query.fvecs"), max_queries);
        const ann::IntMatrix gt = ann::read_ivecs(dir / (name + "_groundtruth.ivecs"), max_queries);
        constexpr std::uint64_t seed = 20261107;
        constexpr std::size_t k = 10;
        const fs::path index_path =
            dir / (name + "_ivf_nlist" + std::to_string(nlist) + "_seed" + std::to_string(seed) + ".bin");

        std::printf("Skup:      %s, base = %zu, upita = %zu, d = %zu (%s)\n", name.c_str(), base.n, queries.n, base.d,
                    build_name());
        if (fs::exists(index_path))
            std::printf("Indeks:    ucitan iz %s\n", index_path.string().c_str());
        else
        {
            const ann::FloatMatrix learn = ann::read_fvecs(dir / (name + "_learn.fvecs"));
            ann::KMeansParams km;
            km.seed = seed;
            ann::IvfFlatSeq builder(ann::l2_scalar, km);
            const auto t0 = std::chrono::steady_clock::now();
            builder.train(learn, nlist);
            builder.add(base);
            ann::save_ivf_lists(builder.lists(), index_path);
            std::printf("Indeks:    napravljen skalarnom verzijom za %.1f s i sacuvan u %s\n", seconds_since(t0),
                        index_path.string().c_str());
        }

        auto load_index = [&](ann::DistFn dist)
        {
            ann::IvfLists lists = ann::load_ivf_lists(index_path);
            if (!ann::ivf_lists_match(lists, nlist, base.d, base.n))
                throw std::runtime_error("indeks " + index_path.string() +
                                         " ne odgovara nlist-u ili bazi; obrisati ga da se napravi ponovo");
            ann::IvfFlatSeq index(dist);
            index.set_lists(std::move(lists));
            return index;
        };

        // Skalarna referenca (za ostale varijante), u svom bloku da se njen
        // indeks oslobodi prije mjerenja.
        ann::SearchResult reference;
        const bool is_reference = variant->fn == ann::l2_scalar;
        const bool compare = check && !is_reference;
        if (compare)
            reference = load_index(ann::l2_scalar).search(queries, k, nprobe);

        const ann::IvfFlatSeq index = load_index(variant->fn);
        ann::SearchResult result;
        std::size_t scanned = 0;
        perf::Config config;
        config.warmups = 1;
        config.repeats = repeats;
        const perf::Sample sample = perf::measure(
            config, [&] { result = index.search(queries, k, nprobe, &scanned); },
            [&] { return perf::checksum(result.distances.data(), result.distances.size()); });

        std::size_t identical = queries.n;
        if (compare)
        {
            identical = 0;
            for (std::size_t q = 0; q < queries.n; ++q)
            {
                const auto from = static_cast<std::ptrdiff_t>(q * k);
                const auto to = from + static_cast<std::ptrdiff_t>(k);
                identical += std::equal(result.ids.begin() + from, result.ids.begin() + to, reference.ids.begin() + from) &&
                                     std::equal(result.distances.begin() + from, result.distances.begin() + to,
                                                reference.distances.begin() + from)
                                 ? 1
                                 : 0;
            }
        }

        const double nq = static_cast<double>(queries.n);
        const double median = sample.median_s;
        const double per_query = static_cast<double>(scanned) / nq;
        // Udaljenosti po upitu: do svih centroida (izbor listi) + skenirani vektori.
        const double distances = static_cast<double>(scanned) + nq * static_cast<double>(index.lists().nlist());
        const double ns_per_dist = median * 1e9 / distances;
        // Propusnost skeniranja: samo vektori iz listi (centroidi su mali i ostaju u kešu).
        const double gbps = static_cast<double>(scanned) * static_cast<double>(base.d) * sizeof(float) / median / 1e9;
        const double recall = ann::recall_at_k(result, gt, k);

        std::printf("Pretraga:  nlist = %zu, nprobe = %zu, k = %zu, udaljenost = %s, %d ponavljanja\n\n", nlist,
                    nprobe, k, variant->name, repeats);
        std::printf("  medijan / minimum:        %.4f s / %.4f s\n", median, sample.min_s);
        std::printf("  QPS:                      %.1f\n", nq / median);
        std::printf("  skenirano po upitu:       %.0f vektora (%.2f %% baze)\n", per_query,
                    100.0 * per_query / static_cast<double>(base.n));
        std::printf("  ns po udaljenosti:        %.2f (ukljucujuci udaljenosti do centroida)\n", ns_per_dist);
        std::printf("  propusnost skeniranja:    %.2f GB/s\n", gbps);
        std::printf("  Recall@10:                %.4f\n", recall);
        if (is_reference)
            std::printf("  identicno skalarnoj:      (ovo je skalarna referenca)\n");
        else if (!compare)
            std::printf("  identicno skalarnoj:      nije provjereno (provjera = 0)\n");
        else
            std::printf("  identicno skalarnoj:      %zu / %zu upita\n", identical, queries.n);
        std::printf("\nCSV: %s,%s,%zu,%zu,%zu,%d,%.6f,%.6f,%.1f,%.0f,%.3f,%.3f,%.4f,%zu\n", variant->name, name.c_str(),
                    nlist, nprobe, queries.n, repeats, median, sample.min_s, nq / median, per_query, ns_per_dist, gbps,
                    recall, identical);
        return identical == queries.n ? 0 : 1;
    }

    std::size_t parse_size(const char* text, const char* name)
    {
        char* end = nullptr;
        const unsigned long long value = std::strtoull(text, &end, 10);
        if (end == text || *end != '\0')
            throw std::invalid_argument(std::string("neispravan broj za ") + name + ": " + text);
        return static_cast<std::size_t>(value);
    }

    // Korak 1: čitač + brute-force kNN moraju se poklopiti sa groundtruth-om.
    int run_gt_check(const fs::path& dir, std::size_t max_queries, std::size_t k)
    {
        const std::string name = fs::path(dir).filename().string();
        const fs::path base_path = dir / (name + "_base.fvecs");
        const fs::path query_path = dir / (name + "_query.fvecs");
        const fs::path gt_path = dir / (name + "_groundtruth.ivecs");

        auto t0 = std::chrono::steady_clock::now();
        const ann::FloatMatrix base = ann::read_fvecs(base_path);
        const ann::FloatMatrix queries = ann::read_fvecs(query_path, max_queries);
        const ann::IntMatrix gt = ann::read_ivecs(gt_path, max_queries);
        const double load_s = seconds_since(t0);

        std::printf("Skup:        %s\n", name.c_str());
        std::printf("Baza:        %zu vektora, d = %zu\n", base.n, base.d);
        std::printf("Upiti:       %zu (d = %zu)\n", queries.n, queries.d);
        std::printf("Groundtruth: %zu upita x %zu susjeda\n", gt.n, gt.d);
        std::printf("Ucitavanje:  %.2f s\n", load_s);

        if (queries.n != gt.n)
        {
            std::fprintf(stderr, "Greska: broj upita (%zu) i redova groundtruth-a (%zu) se razlikuje\n",
                         queries.n, gt.n);
            return 1;
        }
        if (k == 0)
            k = gt.d;

        t0 = std::chrono::steady_clock::now();
        const ann::SearchResult result = ann::bruteforce_knn(base, queries, k, ann::l2_scalar);
        const double search_s = seconds_since(t0);
        std::printf("Brute-force: %.2f s (k = %zu, l2_scalar, %s)\n", search_s, k, build_name());

        const ann::GroundtruthCheck check =
            ann::check_against_groundtruth(result, gt, base, queries, ann::l2_scalar);

        std::printf("\nProvjera do dubine %zu:\n", check.k);
        std::printf("  udaljenosti se poklapaju: %zu / %zu upita\n", check.distances_match, check.queries);
        std::printf("  ID-evi isti, istim redom: %zu / %zu upita (razlika je moguca samo kod jednakih udaljenosti)\n",
                    check.ids_match, check.queries);
        if (k >= 10 && gt.d >= 10)
            std::printf("  Recall@10:                %.4f\n", ann::recall_at_k(result, gt, 10));

        if (!check.passed())
        {
            std::printf("\nNIJE PROSLO. Prvo neslaganje: upit %lld, pozicija %zu: nasa %.1f, ocekivana %.1f\n",
                        static_cast<long long>(check.first_bad_query), check.first_bad_pos,
                        static_cast<double>(check.ours), static_cast<double>(check.expected));
            return 1;
        }
        std::puts("\nPROSLO: brute-force kNN se poklapa sa groundtruth-om.");
        return 0;
    }
}

int main(int argc, char** argv)
{
    try
    {
        if (argc >= 3 && std::string(argv[1]) == "gt-check")
        {
            const std::size_t max_queries = argc >= 4 ? parse_size(argv[3], "broj_upita") : 0;
            const std::size_t k = argc >= 5 ? parse_size(argv[4], "k") : 0;
            return run_gt_check(argv[2], max_queries, k);
        }
        if (argc >= 4 && std::string(argv[1]) == "kmeans-check")
        {
            const std::size_t nlist = parse_size(argv[3], "nlist");
            const int iterations = argc >= 5 ? static_cast<int>(parse_size(argv[4], "iteracije")) : 25;
            const std::uint64_t seed = argc >= 6 ? parse_size(argv[5], "seed") : 20261107;
            return run_kmeans_check(argv[2], nlist, iterations, seed);
        }
        if (argc >= 4 && std::string(argv[1]) == "ivf-check")
        {
            const std::size_t nlist = parse_size(argv[3], "nlist");
            const int iterations = argc >= 5 ? static_cast<int>(parse_size(argv[4], "iteracije")) : 25;
            const std::uint64_t seed = argc >= 6 ? parse_size(argv[5], "seed") : 20261107;
            return run_ivf_check(argv[2], nlist, iterations, seed);
        }
        if (argc >= 4 && std::string(argv[1]) == "ivf-search")
        {
            const std::size_t nlist = parse_size(argv[3], "nlist");
            const std::size_t max_queries = argc >= 5 ? parse_size(argv[4], "broj_upita") : 0;
            const std::size_t k = argc >= 6 ? parse_size(argv[5], "k") : 10;
            return run_ivf_search(argv[2], nlist, max_queries, k);
        }
        if (argc >= 5 && std::string(argv[1]) == "ivf-bench")
        {
            const std::size_t nlist = parse_size(argv[3], "nlist");
            const std::size_t nprobe = parse_size(argv[4], "nprobe");
            const std::size_t max_queries = argc >= 6 ? parse_size(argv[5], "broj_upita") : 1000;
            const int repeats = argc >= 7 ? static_cast<int>(parse_size(argv[6], "ponavljanja")) : 5;
            const std::string dist = argc >= 8 ? argv[7] : "scalar";
            const bool check = argc >= 9 ? parse_size(argv[8], "provjera") != 0 : true;
            return run_ivf_bench(argv[2], nlist, nprobe, max_queries, repeats, dist, check);
        }
        print_usage();
        return 2;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Greska: %s\n", e.what());
        return 1;
    }
}
