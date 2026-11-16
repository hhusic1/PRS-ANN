#include "ivf_io.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ann
{
    namespace
    {
        constexpr char kMagic[8] = { 'P', 'R', 'S', 'I', 'V', 'F', '0', '1' };

        // Gornje granice štite od ogromne alokacije kad je zaglavlje pokvareno
        // (i od prekoračenja pri množenju veličina). SIFT1B bi stao.
        constexpr std::uint64_t kMaxNlist = 1ULL << 24;
        constexpr std::uint64_t kMaxD = 1ULL << 16;
        constexpr std::uint64_t kMaxN = 1ULL << 31;

        template <class T>
        void write_array(std::ofstream& out, const T* data, std::size_t count)
        {
            out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
        }

        template <class T>
        void read_array(std::ifstream& in, T* data, std::size_t count)
        {
            in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
        }
    }

    void save_ivf_lists(const IvfLists& lists, const std::filesystem::path& path)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("ne mogu otvoriti fajl za pisanje: " + path.string());

        const std::uint64_t header[3] = { lists.nlist(), lists.d(), lists.n };
        out.write(kMagic, sizeof(kMagic));
        write_array(out, header, 3);
        write_array(out, lists.centroids.values.data(), lists.nlist() * lists.d());
        for (const std::size_t o : lists.offset)
        {
            const std::uint64_t v = o;
            write_array(out, &v, 1);
        }
        write_array(out, lists.list_data.data(), lists.n * lists.d());
        write_array(out, lists.list_ids.data(), lists.n);
        if (!out)
            throw std::runtime_error("greska pri pisanju fajla: " + path.string());
    }

    IvfLists load_ivf_lists(const std::filesystem::path& path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("ne mogu otvoriti fajl indeksa: " + path.string());
        const std::uintmax_t file_size = std::filesystem::file_size(path);

        char magic[8] = {};
        std::uint64_t header[3] = {};
        in.read(magic, sizeof(magic));
        read_array(in, header, 3);
        if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
            throw std::runtime_error("fajl nije PRS-ANN indeks (pogresan magicni niz): " + path.string());

        const std::uint64_t nlist = header[0], d = header[1], n = header[2];
        if (nlist == 0 || nlist > kMaxNlist || d == 0 || d > kMaxD || n > kMaxN)
            throw std::runtime_error("neispravno zaglavlje indeksa: " + path.string());

        const std::uint64_t expected = sizeof(kMagic) + 3 * sizeof(std::uint64_t) + nlist * d * sizeof(float) +
                                       (nlist + 1) * sizeof(std::uint64_t) + n * d * sizeof(float) +
                                       n * sizeof(std::int32_t);
        if (file_size != expected)
            throw std::runtime_error("velicina fajla indeksa ne odgovara zaglavlju (skracen ili tudji fajl): " +
                                     path.string());

        IvfLists lists;
        lists.centroids.n = nlist;
        lists.centroids.d = d;
        lists.centroids.values = simd::AlignedFloats(nlist * d);
        read_array(in, lists.centroids.values.data(), nlist * d);

        lists.offset.resize(nlist + 1);
        for (std::size_t c = 0; c <= nlist; ++c)
        {
            std::uint64_t v = 0;
            read_array(in, &v, 1);
            lists.offset[c] = v;
        }

        lists.n = n;
        lists.list_data = simd::AlignedFloats(n * d);
        read_array(in, lists.list_data.data(), n * d);
        lists.list_ids.resize(n);
        read_array(in, lists.list_ids.data(), n);
        if (!in)
            throw std::runtime_error("greska pri citanju fajla indeksa: " + path.string());

        bool offset_ok = lists.offset.front() == 0 && lists.offset.back() == n;
        for (std::size_t c = 0; c < nlist && offset_ok; ++c)
            offset_ok = lists.offset[c] <= lists.offset[c + 1];
        if (!offset_ok)
            throw std::runtime_error("offset u fajlu indeksa nije ispravan CSR niz: " + path.string());
        return lists;
    }

    bool ivf_lists_match(const IvfLists& lists, std::size_t nlist, std::size_t d, std::size_t n)
    {
        return lists.nlist() == nlist && lists.d() == d && lists.n == n;
    }

    bool ivf_lists_match_base(const IvfLists& lists, const FloatMatrix& base)
    {
        if (lists.d() != base.d || lists.n != base.n || lists.list_ids.size() != lists.n)
            return false;
        const std::size_t d = base.d;
        for (std::size_t p = 0; p < lists.n; ++p)
        {
            const std::int32_t id = lists.list_ids[p];
            if (id < 0 || static_cast<std::size_t>(id) >= base.n ||
                std::memcmp(lists.list_data.data() + p * d, base.row(static_cast<std::size_t>(id)),
                            d * sizeof(float)) != 0)
                return false;
        }
        return true;
    }
}
