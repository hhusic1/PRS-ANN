#pragma once

// IVF-Flat indeks: invertovane liste sa originalnim (nekomprimovanim) vektorima.
//
// Ideja (Sivic & Zisserman 2003; Jégou, Douze, Schmid 2011):
//  - k-means podijeli prostor na nlist Voronoi ćelija (centroidi),
//  - svaki vektor baze ide u listu svog najbližeg centroida,
//  - pri pretrazi se skeniraju samo liste nprobe najbližih centroida upitu.
//
// Raspored u memoriji je CSR (compressed sparse row), kao u FAISS-u:
//
//   offset    [nlist + 1]  lista c zauzima pozicije [offset[c], offset[c+1])
//   list_data [N × d]      vektori sortirani po listama, jedan kontinuiran niz
//   list_ids  [N]          originalni ID vektora na svakoj poziciji
//
// Zašto CSR, a ne std::vector<std::vector<float>> po listi: jedna alokacija,
// lista je kontinuiran blok memorije (sekvencijalno čitanje, hardverski
// prefetch, SIMD bez skokova), a offset odmah daje dužinu liste.

#include "dataset.hpp"
#include "distance.hpp"
#include "kmeans.hpp"
#include "topk.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ann
{
    struct IvfLists
    {
        FloatMatrix centroids;            // nlist × d
        std::vector<std::size_t> offset;  // nlist + 1, offset[0] = 0, offset[nlist] = N
        simd::AlignedFloats list_data;    // N × d
        std::vector<std::int32_t> list_ids; // N
        std::size_t n = 0;                // broj vektora u indeksu

        std::size_t nlist() const { return centroids.n; }
        std::size_t d() const { return centroids.d; }
        std::size_t list_size(std::size_t c) const { return offset[c + 1] - offset[c]; }
        const float* list_vectors(std::size_t c) const { return list_data.data() + offset[c] * d(); }
        const std::int32_t* list_vector_ids(std::size_t c) const { return list_ids.data() + offset[c]; }
    };

    // Bira nprobe centroida najbližih upitu i upisuje ih u probes[0..nprobe),
    // sortirane rastuće po udaljenosti (Neighbor::id = indeks centroida).
    // Pri jednakoj udaljenosti bira se manji indeks (isti poredak `better` kao
    // u top-k), pa je izbor listi deterministički.
    //
    // std::partial_sort umjesto punog sortiranja: treba samo nprobe najboljih
    // od nlist, O(nlist · log nprobe) umjesto O(nlist · log nlist). Prefiks
    // ostaje sortiran, pa se liste skeniraju od najbliže: heap se brzo napuni
    // dobrim kandidatima, a kasniji (lošiji) se odbacuju jednim poređenjem.
    // Funkciju dijeli i OpenMP verzija (svaka nit ima svoj `probes`).
    void select_probes(const float* query, const FloatMatrix& centroids, DistFn dist,
                       std::size_t nprobe, std::vector<Neighbor>& probes);

    // Sekvencijalna verzija (referenca). OpenMP verzija će imati isti
    // interfejs: train(learn, nlist), add(base), search(queries, k, nprobe).
    class IvfFlatSeq
    {
    public:
        explicit IvfFlatSeq(DistFn dist = l2_scalar, KMeansParams kmeans = {});

        // Trenira centroide k-meansom na learn skupu. Vraća statistiku k-meansa.
        KMeansResult train(const FloatMatrix& learn, std::size_t nlist);

        // Postavlja gotove centroide (umjesto train). Koristi se da OpenMP
        // verzija i FAISS dobiju potpuno iste centroide kao sekvencijalna.
        void set_centroids(const FloatMatrix& centroids);

        // Dodaje bazu u invertovane liste. Poziva se jednom, nakon train.
        void add(const FloatMatrix& base);

        // k najbližih susjeda za svaki upit, skeniranjem nprobe najbližih
        // listi. nprobe > nlist se svodi na nlist (tada je pretraga tačna, kao
        // brute-force). Ako je u skeniranim listama manje od k vektora, ostatak
        // je id = -1, dist = +inf. `scanned` (opciono) dobija ukupan broj
        // skeniranih vektora, za računanje propusnosti u fazi mjerenja.
        SearchResult search(const FloatMatrix& queries, std::size_t k, std::size_t nprobe,
                            std::size_t* scanned = nullptr) const;

        bool is_trained() const { return _lists.centroids.n > 0; }
        const IvfLists& lists() const { return _lists; }

    private:
        DistFn _dist;
        KMeansParams _kmeans;
        IvfLists _lists;
    };
}
