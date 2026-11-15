#pragma once

// Čuvanje IVF indeksa (centroidi + CSR liste) u binarni fajl.
//
// Zašto: mjerenja pretrage (ivf-bench, VTune) moraju se raditi na IDENTIČNOM
// indeksu. Izgradnja na SIFT1M traje minutama, a brža funkcija udaljenosti
// daje centroide koji se razlikuju u zadnjim bitima, pa bi "prije" i
// "poslije" bili različiti indeksi. Indeks se zato jednom napravi skalarnom
// verzijom, sačuva, i svako mjerenje ga učita.
//
// Format (little-endian, kao .fvecs; x86 i Apple M su little-endian):
//   "PRSIVF01"          8 B, magični niz i verzija formata
//   uint64 nlist, d, n  zaglavlje
//   float  centroids    nlist × d
//   uint64 offset       nlist + 1
//   float  list_data    n × d
//   int32  list_ids     n

#include "ivf_index.hpp"

#include <cstddef>
#include <filesystem>

namespace ann
{
    // Baca std::runtime_error ako se fajl ne može upisati.
    void save_ivf_lists(const IvfLists& lists, const std::filesystem::path& path);

    // Baca std::runtime_error ako fajl ne postoji, nema ispravan magični niz,
    // veličina mu ne odgovara zaglavlju (skraćen ili tuđi fajl) ili offset
    // nije ispravan CSR niz.
    IvfLists load_ivf_lists(const std::filesystem::path& path);

    // Da li indeks odgovara traženom nlist i bazi (d, n). Štiti od mjerenja
    // na indeksu napravljenom za drugi nlist ili drugi skup podataka.
    bool ivf_lists_match(const IvfLists& lists, std::size_t nlist, std::size_t d, std::size_t n);
}
