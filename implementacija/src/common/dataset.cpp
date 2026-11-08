#include "dataset.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace ann
{
    namespace
    {
        struct Header
        {
            std::size_t n = 0;
            std::size_t d = 0;
        };

        [[noreturn]] void fail(const std::filesystem::path& path, const std::string& what)
        {
            throw std::runtime_error(path.string() + ": " + what);
        }

        // Dimenzija se čita iz prvog zapisa, a broj vektora iz veličine fajla:
        // svaki zapis ima tačno 4 + d * elem_size bajtova. Ako veličina nije
        // djeljiva sa veličinom zapisa, fajl je oštećen ili nije ovog formata.
        Header inspect(std::ifstream& in, const std::filesystem::path& path,
                       std::size_t elem_size, std::size_t max_rows)
        {
            std::int32_t dim = 0;
            if (!in.read(reinterpret_cast<char*>(&dim), sizeof(dim)))
                fail(path, "prazan fajl ili greska pri citanju");
            if (dim <= 0 || dim > 1'000'000)
                fail(path, "neispravna dimenzija " + std::to_string(dim));

            const std::uintmax_t file_bytes = std::filesystem::file_size(path);
            const std::uintmax_t record_bytes = sizeof(std::int32_t) + static_cast<std::uintmax_t>(dim) * elem_size;
            if (file_bytes % record_bytes != 0)
                fail(path, "velicina fajla nije visekratnik velicine zapisa");

            Header h;
            h.d = static_cast<std::size_t>(dim);
            h.n = static_cast<std::size_t>(file_bytes / record_bytes);
            if (max_rows != 0 && max_rows < h.n)
                h.n = max_rows;

            in.seekg(0);
            return h;
        }

        // Čita red po red i provjerava da svaki zapis ima istu dimenziju.
        template <class T>
        void read_rows(std::ifstream& in, const std::filesystem::path& path, const Header& h, T* out)
        {
            for (std::size_t i = 0; i < h.n; ++i)
            {
                std::int32_t dim = 0;
                in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                if (!in || static_cast<std::size_t>(dim) != h.d)
                    fail(path, "zapis " + std::to_string(i) + " ima pogresnu dimenziju");

                in.read(reinterpret_cast<char*>(out + i * h.d),
                        static_cast<std::streamsize>(h.d * sizeof(T)));
                if (!in)
                    fail(path, "fajl se prekida u zapisu " + std::to_string(i));
            }
        }

        std::ifstream open(const std::filesystem::path& path)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in)
                fail(path, "ne mogu otvoriti fajl");
            return in;
        }
    }

    FloatMatrix read_fvecs(const std::filesystem::path& path, std::size_t max_rows)
    {
        static_assert(sizeof(float) == 4, "fvecs zahtijeva 32-bitni float");
        std::ifstream in = open(path);
        const Header h = inspect(in, path, sizeof(float), max_rows);

        FloatMatrix m;
        m.n = h.n;
        m.d = h.d;
        m.values = simd::AlignedFloats(h.n * h.d);
        read_rows(in, path, h, m.values.data());
        return m;
    }

    IntMatrix read_ivecs(const std::filesystem::path& path, std::size_t max_rows)
    {
        std::ifstream in = open(path);
        const Header h = inspect(in, path, sizeof(std::int32_t), max_rows);

        IntMatrix m;
        m.n = h.n;
        m.d = h.d;
        m.values.resize(h.n * h.d);
        read_rows(in, path, h, m.values.data());
        return m;
    }
}
