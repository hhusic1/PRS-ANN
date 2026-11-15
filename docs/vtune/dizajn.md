# VTune analiza i optimizacija: dizajn

Datum: 15. 11. 2026. Status: odobren.

## Namjera

Izmjeriti Intel VTune Profilerom gdje sekvencijalna IVF-Flat implementacija gubi vrijeme (čekanje na keš i memoriju, zavisnost podataka, grananje), primijeniti optimizaciju na nađeni problem i ponovo izmjeriti. Sve se dokumentuje tako da se može predstaviti asistentu: metode, metrike prije i poslije, i zaključak.

## Ciljevi

- Top-down slika (Retiring, Front-End Bound, Bad Speculation, Back-End Bound → Memory / Core Bound) i memorijske metrike za pretragu na SIFT1M.
- Jedna ciljana optimizacija, izabrana prema nalazu, uz eksperiment koji odvaja uzroke.
- Ponovljeno mjerenje na **identičnom** indeksu, uz provjeru da se rezultati pretrage nisu promijenili.
- Izvještaj u `docs/vtune/` spreman za prezentaciju.

## Ne-ciljevi

- OpenMP paralelizacija (zasebna faza plana).
- Poređenje sa FAISS-om (faza mjerenja).
- Mijenjanje algoritma (IVF ostaje IVF-Flat).

## Ograničenja

- Windows 11, Intel Core i5-12500H (hibridni: P i E jezgra), MSYS2 UCRT64 GCC; MSVC nije podržan.
- VTune hardversko uzorkovanje traži sampling driver (instalira ga korisnik kao administrator).
- Sekvencijalna verzija je referenca: rezultati se ne smiju promijeniti (za SIFT su udaljenosti upit–vektor tačne bez obzira na redoslijed sabiranja, jer su vrijednosti cijeli brojevi a zbir < 2²⁴).

## Pristupi

| Pristup | Prednost | Mana |
|---|---|---|
| A. Profilisati postojeće komande (`ivf-search`, `kmeans-check`) | bez novog koda | izgradnja i pretraga se miješaju; indeks se gradi iznova svaki put, a brža udaljenost daje malo drugačije centroide, pa "prije" i "poslije" nisu na istom indeksu |
| **B. Nova komanda `ivf-bench` sa sačuvanim indeksom (izabrano)** | identičan indeks prije i poslije, profil pokriva samo pretragu, mjerenje kroz `perf::measure` | ~200 linija novog koda |
| C. Izolovan mikrobenchmark funkcije udaljenosti | najjednostavnije | ne pokazuje stvarno ponašanje keša na listama od 516 MB |

## Dizajn

1. **Čuvanje indeksa:** indeks (centroidi, `offset`, `list_data`, `list_ids`) se jednom napravi skalarnom verzijom i sačuva u binarni fajl; `ivf-bench` ga učitava (ili pravi i čuva ako ne postoji).
2. **`ivf-bench <folder> <nlist> <nprobe> [broj_upita] [ponavljanja] [udaljenost]`:** mjeri samo pretragu (zagrijavanje, ponavljanja, medijan), ispisuje QPS, ns po udaljenosti i propusnost skeniranja (GB/s), i provjerava da je rezultat identičan skalarnoj referenci.
3. **VTune prije optimizacije:** `hotspots`, `uarch-exploration`, `memory-access` na SIFT1M; proces vezan za P-jezgro.
4. **Optimizacija prema nalazu.** Ako se potvrdi hipoteza (lanac zavisnih sabiranja u `l2_scalar`), eksperiment u četiri varijante: skalarna sa 1 akumulatorom (sada), skalarna sa 4 akumulatora, AVX2 + FMA sa 1 akumulatorom, AVX2 + FMA sa 4 akumulatora. Obje AVX2 varijante koriste FMA, pa se razlikuju samo po broju akumulatora. Svaka u svom fajlu, sa svojim zastavicama (kao u Lab 1).
5. **VTune poslije optimizacije**, isti skup analiza.
6. **Izvještaj:** `docs/vtune/izvjestaj.md` sa tabelama, objašnjenjem metrika i zaključcima; sirovi CSV izvještaji VTune-a u `docs/vtune/rezultati/`.

## Prepušteno izvođaču

Tačan format fajla indeksa, izbor `nlist`/`nprobe`/broja upita za profilisanje (cilj: pretraga koja traje dovoljno dugo za VTune, a skenira više nego što staje u keš), i način vezivanja procesa za P-jezgro.
