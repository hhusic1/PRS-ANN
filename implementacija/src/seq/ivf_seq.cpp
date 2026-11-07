#include "ivf_index.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ann
{
    IvfFlatSeq::IvfFlatSeq(DistFn dist, KMeansParams kmeans)
        : _dist(dist), _kmeans(kmeans)
    {
        _kmeans.dist = dist;
    }

    KMeansResult IvfFlatSeq::train(const FloatMatrix& learn, std::size_t nlist)
    {
        KMeansParams params = _kmeans;
        params.nlist = nlist;
        KMeansResult km = kmeans_train_seq(learn, params);

        // Centroidi se premještaju u indeks (bez kopiranja); statistika ostaje
        // pozivaocu, a km.centroids je nakon ovoga prazan.
        _lists = IvfLists{};
        _lists.centroids = std::move(km.centroids);
        return km;
    }

    void IvfFlatSeq::set_centroids(const FloatMatrix& centroids)
    {
        _lists = IvfLists{};
        _lists.centroids.n = centroids.n;
        _lists.centroids.d = centroids.d;
        _lists.centroids.values = simd::AlignedFloats(centroids.n * centroids.d);
        std::memcpy(_lists.centroids.values.data(), centroids.values.data(),
                    centroids.n * centroids.d * sizeof(float));
    }

    // Punjenje listi je "counting sort" po broju liste, u tri prolaza:
    //
    //  1. dodjela:  label[i] = najbliži centroid vektora i   (O(N · nlist · d), skoro svo vrijeme)
    //  2. brojanje: count[c] = broj vektora u listi c, pa offset = prefix sum
    //  3. kopiranje: vektor i ide na poziciju cursor[label[i]]++
    //
    // Vektori se obilaze rastućim ID-em, pa su ID-evi unutar svake liste
    // sortirani rastuće. Taj redoslijed je deterministički, i OpenMP verzija
    // ga mora sačuvati (lokalni brojači po niti + prefix sum preko niti), da bi
    // obje verzije imale bit-identične liste.
    void IvfFlatSeq::add(const FloatMatrix& base)
    {
        if (!is_trained())
            throw std::logic_error("IvfFlatSeq::add: indeks nije treniran (pozvati train ili set_centroids)");
        if (_lists.n != 0)
            throw std::logic_error("IvfFlatSeq::add: baza je vec dodana (podrzano je jedno dodavanje)");
        if (base.d != _lists.d())
            throw std::invalid_argument("IvfFlatSeq::add: dimenzija baze se razlikuje od dimenzije centroida");
        if (base.n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::invalid_argument("IvfFlatSeq::add: ID-evi su int32 (kao u groundtruth.ivecs)");

        const std::size_t n = base.n;
        const std::size_t d = base.d;
        const std::size_t nlist = _lists.nlist();

        // 1. Dodjela.
        std::vector<std::int32_t> labels(n);
        for (std::size_t i = 0; i < n; ++i)
            labels[i] = static_cast<std::int32_t>(nearest_centroid(base.row(i), _lists.centroids, _dist));

        // 2. Brojanje i prefix sum (offset[c] = zbir dužina listi prije c).
        std::vector<std::size_t> counts(nlist, 0);
        for (std::size_t i = 0; i < n; ++i)
            ++counts[static_cast<std::size_t>(labels[i])];

        _lists.offset.assign(nlist + 1, 0);
        for (std::size_t c = 0; c < nlist; ++c)
            _lists.offset[c + 1] = _lists.offset[c] + counts[c];

        // 3. Kopiranje vektora i ID-eva na njihove pozicije.
        _lists.list_data = simd::AlignedFloats(n * d);
        _lists.list_ids.assign(n, -1);
        std::vector<std::size_t> cursor(_lists.offset.begin(), _lists.offset.end() - 1);
        float* out = _lists.list_data.data();
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::size_t pos = cursor[static_cast<std::size_t>(labels[i])]++;
            std::memcpy(out + pos * d, base.row(i), d * sizeof(float));
            _lists.list_ids[pos] = static_cast<std::int32_t>(i);
        }
        _lists.n = n;
    }

    void select_probes(const float* query, const FloatMatrix& centroids, DistFn dist,
                       std::size_t nprobe, std::vector<Neighbor>& probes)
    {
        const std::size_t nlist = centroids.n;
        probes.resize(nlist);
        for (std::size_t c = 0; c < nlist; ++c)
            probes[c] = Neighbor{ dist(query, centroids.row(c), centroids.d), static_cast<std::int32_t>(c) };
        nprobe = std::min(nprobe, nlist);
        std::partial_sort(probes.begin(), probes.begin() + static_cast<std::ptrdiff_t>(nprobe), probes.end(), better);
        probes.resize(nprobe);
    }

    // Pretraga jednog upita:
    //  1. udaljenost upita do svih nlist centroida, izbor nprobe najbližih,
    //  2. skeniranje tih listi (kontinuirani blokovi u list_data) kroz top-k heap.
    //
    // Rezultat ne zavisi od redoslijeda skeniranja listi: top-k pod strogim
    // potpunim poretkom `better` je jedinstven skup. Zato nprobe == nlist daje
    // TAČNO isti rezultat (ID-eve i udaljenosti) kao brute-force, a OpenMP
    // verzija (paralelno po upitima) mora dati bit-identičan rezultat.
    SearchResult IvfFlatSeq::search(const FloatMatrix& queries, std::size_t k, std::size_t nprobe,
                                    std::size_t* scanned) const
    {
        if (!is_trained() || _lists.offset.empty())
            throw std::logic_error("IvfFlatSeq::search: indeks nije treniran ili baza nije dodana");
        if (queries.d != _lists.d())
            throw std::invalid_argument("IvfFlatSeq::search: dimenzija upita se razlikuje od dimenzije indeksa");
        if (k == 0 || nprobe == 0)
            throw std::invalid_argument("IvfFlatSeq::search: k i nprobe moraju biti > 0");

        const std::size_t d = _lists.d();
        SearchResult result(queries.n, k);
        TopK heap(k);
        std::vector<Neighbor> probes;
        std::size_t total_scanned = 0;

        for (std::size_t q = 0; q < queries.n; ++q)
        {
            const float* x = queries.row(q);
            select_probes(x, _lists.centroids, _dist, nprobe, probes);

            for (const Neighbor& probe : probes)
            {
                const std::size_t c = static_cast<std::size_t>(probe.id);
                const float* vectors = _lists.list_vectors(c);
                const std::int32_t* ids = _lists.list_vector_ids(c);
                const std::size_t size = _lists.list_size(c);
                for (std::size_t j = 0; j < size; ++j)
                    heap.push(_dist(x, vectors + j * d, d), ids[j]);
                total_scanned += size;
            }
            heap.write_sorted(result.ids.data() + q * k, result.distances.data() + q * k);
        }

        if (scanned != nullptr)
            *scanned = total_scanned;
        return result;
    }
}
