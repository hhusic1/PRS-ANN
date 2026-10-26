# Paralelni ANN indeks zasnovan na invertovanim listama (IVF)

Projekat iz predmeta **Paralelni računarski sistemi**.

Implementiramo ANN (Approximate Nearest Neighbor) indeks zasnovan na invertovanim listama (IVF-Flat), prvo sekvencijalno, a zatim paralelno na CPU-u (OpenMP i SIMD/AVX2) i GPU-u (CUDA). Mjerimo ubrzanje svake verzije uz očuvanje tačnosti (recall).

LINK ZA GOOGLE DRIVE SA RADOVIMA: https://drive.google.com/drive/folders/1vpXYMi4WDbXk22DF6Yw7slJuBoYYY_VG?usp=drive_link

## Tim

| Član | Indeks |
| --- | --- |
| Haris Husić | 2524/19662 |
| Kerim Hajdar | 2505/19556 |
| Haris Macić | 2511/19555 |

## O projektu

ANN pretraga traži k vektora najbližih upitu, ali odustaje od garancije savršene tačnosti u zamjenu za veliko ubrzanje. IVF indeks radi u dva koraka:

1. **Izgradnja:** k-means dijeli prostor na `nlist` ćelija, a svaki vektor se smješta u invertovanu listu svog najbližeg centroida.
2. **Pretraga:** za upit se pronađe `nprobe` najbližih centroida i pretraže se samo vektori u njihovim listama.

Parametar `nprobe` određuje kompromis između brzine i tačnosti. Paralelizujemo i izgradnju indeksa i pretragu.

## Verzije

| Verzija | Tehnologija |
| --- | --- | --- |
| Brute-force kNN | C++ (referenca za recall) |
| Sekvencijalni IVF-Flat | C++ | 
| SIMD | Auto-vektorizacija, `omp simd`, AVX2 + FMA |
| OpenMP | `parallel for`, redukcije, dinamičko raspoređivanje |
| OpenMP + SIMD | Kombinacija |
| CUDA | Kerneli za k-means, scan, pretragu, top-k (Thrust) |

## Struktura repozitorija

```
.
├── CMakeLists.txt
├── include/            # zaglavlja (interfejs indeksa, udaljenosti, mjerenje)
├── src/
│   ├── common/         # učitavanje podataka, recall, mjerni okvir
│   ├── distance/       # l2_scalar, l2_auto, l2_omp, l2_avx2
│   ├── seq/            # sekvencijalni k-means i IVF
│   ├── omp/            # OpenMP verzija
│   └── main.cpp
├── cuda/               # CUDA verzija (.cu fajlovi)
├── notebooks/          # Google Colab notebook za CUDA
├── scripts/            # preuzimanje podataka, FAISS referenca, grafici
├── results/            # izmjereni rezultati (CSV) i grafici
├── docs/               # plan, literatura, izvještaj, prezentacija
└── data/               # skupovi podataka (nisu u repozitoriju)
```

## Zahtjevi

- C++ kompajler sa podrškom za C++20 ili noviji (g++ 11+ ili clang 14+)
- CMake 3.16+
- OpenMP
- Procesor sa AVX2 i FMA podrškom (za SIMD verziju)
- NVIDIA GPU i CUDA Toolkit 12+ (za CUDA verziju), ili Google Colab
- Python 3 sa `numpy`, `matplotlib` i `faiss-cpu` (za grafike i referentna mjerenja)

## Podaci

Koristimo skupove SIFT10K (razvoj) i SIFT1M (konačna mjerenja) sa stranice [corpus-texmex.irisa.fr](http://corpus-texmex.irisa.fr/). Fajlovi su u formatu `.fvecs` / `.ivecs` i sadrže bazne vektore, upite i tačne odgovore.

```bash
./scripts/download_sift.sh      # preuzima i raspakuje podatke u data/
```

Podaci se ne čuvaju u repozitoriju zbog veličine.

## Build i pokretanje

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/ann_ivf --data data/siftsmall --nlist 100 --nprobe 8 --k 10 --threads 4
```

CMake opcije:

| Opcija | Podrazumijevano | Opis |
| --- | --- | --- |
| `ANN_ENABLE_OPENMP` | `ON` | OpenMP niti i `omp simd` |
| `ANN_ENABLE_AVX2` | `ON` | AVX2 + FMA verzija funkcije udaljenosti |
| `ANN_VECTORIZATION_REPORTS` | `OFF` | Izvještaji kompajlera o vektorizaciji |

CUDA verzija se pokreće kroz `notebooks/colab_cuda.ipynb` na Google Colabu.

## Evaluacija

Sve verzije se mjere na istom skupu podataka, istim parametrima i istim mjernim okvirom (zagrijavanje, više ponavljanja, medijan).

| Metrika | Opis |
| --- | --- |
| Recall@10 | Udio pravih 10 najbližih susjeda koji su pronađeni |
| QPS | Broj upita u sekundi |
| Vrijeme izgradnje | K-means i punjenje invertovanih listi |
| Ubrzanje i efikasnost | U odnosu na sekvencijalnu verziju, za 1, 2, 4 i 8 niti |

Eksperimenti: `nprobe` ∈ {1, 8, 32, 128}, niti ∈ {1, 2, 4, 8}, poređenje sa FAISS `IndexIVFFlat` na CPU-u.

## Rezultati

_Biće dodani nakon mjerenja._ Sirovi rezultati su u `results/`, a grafici u `results/figures/`.

## Literatura

- H. Jégou, M. Douze, C. Schmid, [Product Quantization for Nearest Neighbor Search](https://inria.hal.science/inria-00514462v2), IEEE TPAMI, 2011.
- J. Sivic, A. Zisserman, [Video Google: A Text Retrieval Approach to Object Matching in Videos](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/sivic03.pdf), ICCV, 2003.
- A. Babenko, V. Lempitsky, [The Inverted Multi-Index](http://sites.skoltech.ru/app/data/uploads/sites/25/2014/12/TPAMI14.pdf), IEEE TPAMI, 2015.
- D. Baranchuk, A. Babenko, Y. Malkov, [Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors](https://arxiv.org/abs/1802.02422), ECCV, 2018.
- J. Johnson, M. Douze, H. Jégou, [Billion-Scale Similarity Search with GPUs](https://arxiv.org/abs/1702.08734), IEEE Transactions on Big Data, 2019.
- M. Douze et al., [The Faiss Library](https://arxiv.org/abs/2401.08281), 2024.

Potpuna lista literature je u `docs/literatura.md`.

## Izvori koda

- Sekvencijalna IVF implementacija je zasnovana na [vector-db-from-scratch](https://github.com/skyzh/vector-db-from-scratch) (Alex Chi Z, licenca CC BY-NC-SA 4.0) i prilagođena za ovaj projekat.
- Mjerni okvir i SIMD obrazac su preuzeti iz laboratorijske vježbe 1 (Uvod i vektorizacija) predmeta Paralelni računarski sistemi.
- Referentna mjerenja koriste biblioteku [FAISS](https://github.com/facebookresearch/faiss).
