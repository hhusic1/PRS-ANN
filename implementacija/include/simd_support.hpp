#pragma once

// Preuzeto iz Lab 1 (lab1/include/simd_support.hpp), sa izmjenama:
//  1. Na Windowsu (MinGW/UCRT, ne samo MSVC) koristi se _aligned_malloc, jer
//     posix_memalign tamo ne postoji. Na Linuxu/macOS-u ostaje posix_memalign.
//  2. Dodani su move konstruktor/dodjela i size(), da bi se skupovi podataka
//     mogli vraćati iz funkcija (kopiranje ostaje zabranjeno: N × d floatova
//     se nikad ne kopira slučajno).
//  3. Podrazumijevani konstruktor pravi prazan niz.

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace simd
{
    // 32 B = širina AVX2 registra (8 floatova). Poravnat početak niza znači da
    // je i svaki red od d = 128 floatova (512 B) poravnat.
    inline constexpr std::size_t kAlignment = 32;

    class AlignedFloats
    {
    private:
        float* _data = nullptr;
        std::size_t _size = 0;

        static float* allocate(std::size_t count)
        {
            if (count == 0)
                return nullptr;
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
                throw std::bad_alloc();

            const std::size_t bytes = count * sizeof(float);
            void* ptr = nullptr;
#if defined(_WIN32)
            ptr = _aligned_malloc(bytes, kAlignment);
#else
            if (posix_memalign(&ptr, kAlignment, bytes) != 0)
                ptr = nullptr;
#endif
            if (!ptr)
                throw std::bad_alloc();
            return static_cast<float*>(ptr);
        }

        static void release(float* ptr)
        {
#if defined(_WIN32)
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }

    public:
        AlignedFloats() = default;

        explicit AlignedFloats(std::size_t count)
            : _data(allocate(count)), _size(count)
        {
        }

        ~AlignedFloats()
        {
            release(_data);
        }

        AlignedFloats(const AlignedFloats&) = delete;
        AlignedFloats& operator=(const AlignedFloats&) = delete;

        AlignedFloats(AlignedFloats&& other) noexcept
            : _data(std::exchange(other._data, nullptr)),
              _size(std::exchange(other._size, 0))
        {
        }

        AlignedFloats& operator=(AlignedFloats&& other) noexcept
        {
            if (this != &other)
            {
                release(_data);
                _data = std::exchange(other._data, nullptr);
                _size = std::exchange(other._size, 0);
            }
            return *this;
        }

        float* data() const
        {
            return _data;
        }

        std::size_t size() const
        {
            return _size;
        }
    };

    inline void check_vector(const std::vector<float>& expected, const float* actual,
                             std::size_t n, float absolute_tolerance = 1e-4f)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > absolute_tolerance * (1.0f + std::fabs(expected[i])))
                throw std::runtime_error("SIMD rezultat nije tacan na indeksu: " + std::to_string(i));
        }
    }
}
