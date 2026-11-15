#include "distance.hpp"

// Registar varijanti udaljenosti. Kompajlira se BEZ -mavx2: provjera procesora
// mora raditi i na procesoru koji AVX2 nema.

namespace ann
{
    const std::vector<DistanceVariant>& distance_variants()
    {
        static const std::vector<DistanceVariant> variants = {
            { "scalar", l2_scalar, false },
        };
        return variants;
    }

    const DistanceVariant* find_distance(std::string_view name)
    {
        for (const DistanceVariant& v : distance_variants())
            if (name == v.name)
                return &v;
        return nullptr;
    }

    bool cpu_has_avx2_fma()
    {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        return false;
#endif
    }
}
