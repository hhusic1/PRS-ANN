# PRS-ANN – implementacija

IVF-Flat indeks za približnu pretragu najbližih susjeda (Approximate Nearest Neighbor), u C++20:

1. sekvencijalna verzija (referenca) – **gotova i provjerena**,
2. SIMD verzija funkcije udaljenosti (AVX2 + FMA) – sljedeće,
3. OpenMP verzija (izgradnja indeksa i pretraga),
4. mjerenja i poređenje sa FAISS `IndexIVFFlat` (CPU).

Ovaj dokument opisuje kako projekat postaviti na novom računaru (Windows, macOS, Linux),
kako ga pokrenuti, šta se provjerava i kako čitati rezultate.

---

## Sadržaj

1. [Šta je potrebno](#1-šta-je-potrebno)
2. [Instalacija alata](#2-instalacija-alata) – [Windows](#windows) · [macOS](#macos) · [Linux](#linux)
3. [Preuzimanje koda](#3-preuzimanje-koda)
4. [Podaci (SIFT)](#4-podaci-sift)
5. [Build](#5-build)
6. [Provjera da sve radi](#6-provjera-da-sve-radi)
7. [Komande i kako čitati rezultate](#7-komande-i-kako-čitati-rezultate)
8. [Struktura projekta](#8-struktura-projekta)
9. [Python (faza mjerenja)](#9-python-faza-mjerenja)
10. [IDE](#10-ide)
11. [Česti problemi](#11-česti-problemi)

---

## 1. Šta je potrebno

| Šta | Minimalno | Napomena |
|---|---|---|
| C++ kompajler | GCC ≥ 11 ili Clang ≥ 14 (C++20) | na Windowsu **samo GCC iz MSYS2**, ne MSVC |
| CMake | ≥ 3.20 | |
| Ninja | bilo koja novija | generator za build |
| git | | |
| SIFT podaci | SIFT10K (~5 MB), SIFT1M (~160 MB arhiva) | nisu na GitHubu, preuzimaju se posebno |
| Python 3.12 + numpy, matplotlib, faiss-cpu | | tek za fazu mjerenja |

Provjereno okruženje (na kome su rađena sva mjerenja): Windows 11, MSYS2 UCRT64,
GCC 16.2, CMake 4.4, Ninja 1.13, procesor x86-64 sa AVX2 (12 logičkih jezgara).

---

## 2. Instalacija alata

### Windows

> **MSVC (Visual Studio kompajler) nije podržan** – podržava samo OpenMP 2.0 (bez
> `omp simd reduction`). CMake ga namjerno odbija porukom `MSVC nije podržan`.

1. Instalirati MSYS2 (PowerShell):
   ```powershell
   winget install MSYS2.MSYS2
   ```
2. Otvoriti **„MSYS2 UCRT64"** iz Start menija i ažurirati sistem. Ponavljati dok ne kaže
   da nema ništa za ažuriranje (prozor se može sam zatvoriti usred ažuriranja – ponovo ga otvoriti):
   ```bash
   pacman -Syu
   ```
3. U istom prozoru instalirati alate:
   ```bash
   pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
   ```
4. Dodati `C:\msys64\ucrt64\bin` u PATH (Postavke → Sistem → About → Advanced system
   settings → Environment Variables → `Path` → New), pa otvoriti **novi** PowerShell.
5. Provjera:
   ```powershell
   g++ --version; cmake --version; ninja --version
   ```

Pravila za Windows:
- MSYS2 ažurirati **samo** potpunim `pacman -Syu`. `pacman -S paket` na neažuriranu bazu
  pravi djelimično ažuriranje i konflikte paketa.
- Putanja projekta **bez razmaka i naših slova** (dobro: `C:\Users\Ime\source\repos\PRS-ANN`,
  loše: `C:\Users\Ime\Radna površina\PRS ANN`).
- Programi se na Windowsu linkuju statički (`-static`), pa ne zavise od DLL-ova u PATH-u
  (vidi [Česti problemi](#11-česti-problemi)).

### macOS

1. Apple alati (Apple clang i git):
   ```bash
   xcode-select --install
   ```
2. [Homebrew](https://brew.sh), pa CMake i Ninja:
   ```bash
   brew install cmake ninja
   ```
3. Za OpenMP fazu (kasnije) – Apple clang nema OpenMP biblioteku:
   ```bash
   brew install libomp
   ```

Napomene za Mac sa M-čipom (ARM):
- `CMakeLists.txt` sam bira `-mcpu=native` umjesto `-march=native` (x86 zastavica).
- AVX2 verzija (SIMD faza) na ARM-u **ne postoji** – to su Intel/AMD instrukcije. Skalarna,
  auto-vektorizovana i `omp simd` verzija rade.
- Sva mjerenja za izvještaj rade se na jednom x86 računaru, da brojevi budu uporedivi.
- Ne pokretati CMake kroz Rosettu (x86 Terminal) – tada se računar predstavlja kao x86_64.

### Linux

Ubuntu / Debian:
```bash
sudo apt install g++ cmake ninja-build git
```
OpenMP je uz GCC već uključen (`libgomp`).

---

## 3. Preuzimanje koda

```bash
git clone https://github.com/hhusic1/PRS-ANN.git
cd PRS-ANN/implementacija
```

Dok sekvencijalna verzija nije spojena u `main`, raditi na grani `feature/seq`:
```bash
git checkout feature/seq
```

Sve komande ispod se pokreću **iz foldera `implementacija/`**.

---

## 4. Podaci (SIFT)

Podaci su u `.gitignore` i svako ih preuzima sam sa
[corpus-texmex.irisa.fr](http://corpus-texmex.irisa.fr/):

| Skup | Arhiva | Base | Query | Learn | Groundtruth | Namjena |
|---|---|---|---|---|---|---|
| ANN_SIFT10K | `siftsmall.tar.gz` (~5 MB) | 10.000 | 100 | 25.000 | 100 NN | razvoj, brze provjere |
| ANN_SIFT1M | `sift.tar.gz` (~160 MB) | 1.000.000 | 10.000 | 100.000 | 100 NN | konačna mjerenja |

Svi vektori imaju d = 128 (SIFT deskriptori, cijeli brojevi 0–255 zapisani kao float).

Raspakovati u `data/` (putanju do arhive prilagoditi; na Windowsu `tar` radi i iz PowerShella,
ili koristiti 7-Zip):
```bash
mkdir -p data
tar -xzf ~/Downloads/siftsmall.tar.gz -C data
tar -xzf ~/Downloads/sift.tar.gz -C data
```

Raspored **mora** biti tačno ovakav (program ime skupa uzima iz imena foldera):
```
data/siftsmall/siftsmall_base.fvecs
data/siftsmall/siftsmall_query.fvecs
data/siftsmall/siftsmall_learn.fvecs
data/siftsmall/siftsmall_groundtruth.ivecs
data/sift/sift_base.fvecs          (i sift_query, sift_learn, sift_groundtruth)
```

Format `.fvecs` / `.ivecs`: za svaki vektor prvo `int32` dimenzija, pa d vrijednosti
(`float` ili `int32`).

---

## 5. Build

Dva builda:
- **Release** (`-O3 -march=native`) – optimizovan; **jedino njegova vremena imaju smisla**.
- **Debug** (`-O0 -g`) – bez optimizacije; za provjeru ispravnosti i debugger.

Konfiguracija (jednom, ili kad se promijeni `CMakeLists.txt`):
```bash
cmake -S . -B build/Release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build/Debug   -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

Kompajliranje (nakon svake promjene koda):
```bash
cmake --build build/Release
cmake --build build/Debug
```

Rezultat (`build/Release/` i `build/Debug/`):
- `prs_ann` – glavni program sa komandama (na Windowsu `prs_ann.exe`),
- `ann_tests` – testovi graničnih slučajeva.

Ako se build pokvari (npr. nakon promjene kompajlera), obrisati folder `build/` i ponoviti konfiguraciju.

---

## 6. Provjera da sve radi

Na Windowsu (PowerShell) programi se pozivaju sa `.\build\Release\prs_ann.exe`, a na
macOS-u / Linuxu sa `./build/Release/prs_ann`. Ispod je oblik za macOS / Linux.

**1. Testovi** – zadnja linija mora biti `SVI TESTOVI PROSLI (0 gresaka)`:
```bash
./build/Release/ann_tests
```

**2. Pretraga na SIFT10K** (nekoliko sekundi):
```bash
./build/Release/prs_ann ivf-search data/siftsmall 64
```

Rezultat mora biti **tačno** ovakav (recall i broj tačnih upita; vremena se razlikuju):

| nprobe | Recall@10 | tacan top-k |
|---|---|---|
| 1 | 0.5630 | 14/100 |
| 8 | 0.9770 | 85/100 |
| 32 | 1.0000 | 100/100 |
| 64 | 1.0000 | 100/100 |

i na kraju `PROSLO: sa nprobe == nlist pretraga je tacna.`

Kod je napisan da bude deterministički na svim sistemima (vlastiti generator slučajnih brojeva
SplitMix64, `-ffp-contract=off`, tačne udaljenosti za SIFT, fiksno pravilo za jednake udaljenosti).
**Ako se brojevi razlikuju na drugom računaru, to je bug** – javiti ostatku tima.

**3. Sve četiri provjere odjednom** (PowerShell):
```powershell
foreach ($cmd in @('gt-check data/siftsmall','kmeans-check data/siftsmall 64','ivf-check data/siftsmall 64','ivf-search data/siftsmall 64')) { $a = $cmd -split ' '; .\build\Release\prs_ann.exe @a | Out-Null; "{0,-32} {1}" -f $cmd, $(if ($LASTEXITCODE -eq 0) {'PROSLO'} else {"GRESKA (kod $LASTEXITCODE)"}) }
```
macOS / Linux:
```bash
for c in "gt-check data/siftsmall" "kmeans-check data/siftsmall 64" "ivf-check data/siftsmall 64" "ivf-search data/siftsmall 64"; do ./build/Release/prs_ann $c > /dev/null && echo "PROSLO  $c" || echo "GRESKA  $c"; done
```

Svaka komanda vraća izlazni kod 0 ako je provjera prošla, 1 ako nije.

---

## 7. Komande i kako čitati rezultate

| Komanda | Korak | Trajanje (Release, na punjaču) |
|---|---|---|
| `gt-check <folder> [broj_upita] [k]` | 1: čitač + brute-force | SIFT10K < 1 s; SIFT1M, 100 upita ≈ 25 s |
| `kmeans-check <folder> <nlist> [iteracije] [seed]` | 2: k-means | SIFT10K nlist 64 ≈ 4 s; SIFT1M nlist 256 ≈ 60 s |
| `ivf-check <folder> <nlist> [iteracije] [seed]` | 3: invertovane liste | SIFT1M nlist 256 ≈ 1,5 min |
| `ivf-search <folder> <nlist> [broj_upita] [k]` | 4: pretraga | SIFT10K sekunde; SIFT1M, 1000 upita ≈ 4 min |

Podrazumijevano: svi upiti, k = 100 (gt-check) / 10 (ivf-search), 25 iteracija, seed 20261107.
Na laptopu na bateriji sve traje ~2,5× duže (procesor smanji takt) – rezultati su isti, vremena nisu.

Rezultat se može sačuvati u fajl i istovremeno vidjeti:
```powershell
.\build\Release\prs_ann.exe ivf-search data/siftsmall 128 | Tee-Object results\seq_siftsmall_nlist128.txt
```

### Korak 1 – `gt-check data/siftsmall`

Brute-force (poređenje upita sa svim vektorima) mora dati isto što i zvanični groundtruth.

```
udaljenosti se poklapaju: 100 / 100 upita
ID-evi isti, istim redom: 91 / 100 upita
Recall@10:                1.0000
PROSLO
```
- **Udaljenosti po pozicijama** su glavni kriterij: za svaki upit i svaku poziciju j udaljenost
  našeg j-tog susjeda == udaljenost j-tog susjeda iz groundtruth-a. Poređenje je na tačnu
  jednakost, jer je za SIFT udaljenost u floatu tačna (cijeli brojevi, zbir < 2²⁴).
- **ID-evi** mogu da se razlikuju (91/100, a na SIFT1M 53/100) i to **nije greška**: kad dva vektora
  imaju istu udaljenost, groundtruth uzme jedan, a mi uvijek manji ID. Oba rezultata su tačna.

### Korak 2 – `kmeans-check data/siftsmall 64`

- Tabela po iteracijama: ciljna funkcija (zbir kvadrata udaljenosti tačaka do centroida) mora
  **opadati** (npr. 2.785.350.951 → 1.647.073.919). Relativna promjena je negativna i sve manja.
- Kolona `prazni`: koliko praznih klastera je popravljeno (na SIFT-u uvijek 0).
- **Imbalance faktor** `nlist · Σ|Lᵢ|² / N²`: 1,0 = sve liste iste dužine; na SIFT-u ≈ 1,1.

### Korak 3 – `ivf-check data/siftsmall 64`

Šest provjera, sve moraju biti `OK`:
1. `offset[0] = 0`, `offset[nlist] = N`, offset ne opada,
2. zbir dužina listi == N,
3. svaki ID se javlja tačno jednom,
4. ID-evi rastu unutar svake liste (deterministički raspored),
5. vektor u listi == originalni vektor iz baze,
6. vektor je u listi svog najbližeg centroida (za N > 100.000 na uzorku od 10.000).

Statistika: SIFT10K nlist 64 – liste 21–387 vektora (prosjek 156), imbalance 1,195, 5,2 MB;
SIFT1M nlist 256 – liste 1026–10256 (prosjek 3906), imbalance 1,095, 516 MB.

### Korak 4 – `ivf-search data/siftsmall 64`

```
 nprobe  Recall@10  tacan top-k  skenirano/upit (% baze)   vrijeme    QPS
      1    0.5630      14/100        205  ( 2.05 %)          ...      ...
      8    0.9770      85/100       1441  (14.41 %)
     32    1.0000     100/100       5091  (50.91 %)
     64    1.0000     100/100      10000 (100.00 %)
```

| Kolona | Značenje |
|---|---|
| `nprobe` | koliko najbližih listi se skenira po upitu |
| `Recall@10` | udio pravih 10 najbližih koje smo našli (1,0 = svi) |
| `tacan top-k` | broj upita kod kojih je svih k rezultata tačno (strože od recall-a) |
| `skenirano/upit` | sa koliko vektora je upit prosječno poređen – određuje cijenu pretrage |
| `vrijeme`, `QPS` | ukupno vrijeme i upiti u sekundi (orijentaciono, jedno pokretanje) |

Šta se vidi:
- **Kompromis IVF-a:** veći `nprobe` → veći recall, manji QPS.
- Sa `nprobe = 1` recall je nizak jer su pravi susjedi često u susjednoj ćeliji (upit blizu granice).
- Vrijeme je približno proporcionalno broju skeniranih vektora.
- Skenira se više od `nprobe · N / nlist` jer upiti češće padaju u veće liste (imbalance).

Uslov prolaza (za `nprobe == nlist`): udaljenosti == groundtruth na svim pozicijama **i**
rezultat identičan brute-force-u ID po ID (za N > 100.000 prvih 100 upita).

SIFT1M, nlist 256, 1000 upita, jedna nit (orijentaciono):

| nprobe | 1 | 8 | 32 | 128 | 256 |
|---|---|---|---|---|---|
| Recall@10 | 0,472 | 0,932 | 0,9975 | 0,9994 | 0,9994 |
| skenirano | 0,43 % | 3,4 % | 13 % | 49 % | 100 % |
| QPS | 2286 | 314 | 83 | 22 | 11 |

Recall sa `nprobe = nlist` je 0,9994, a ne 1,0, zbog jednakih udaljenosti (kao u koraku 1);
udaljenosti se poklapaju na svih 1000 upita.

---

## 8. Struktura projekta

```
implementacija/
├─ CMakeLists.txt     build: fajlovi, zastavice po fajlu, programi
├─ README.md          ovaj dokument
├─ include/           zaglavlja – ŠTA postoji (strukture, deklaracije)
├─ src/
│  ├─ common/         čitanje podataka, ocjena kvaliteta, mjerni okvir
│  ├─ distance/       funkcije udaljenosti (svaka u svom fajlu, sa svojim zastavicama)
│  ├─ seq/            sekvencijalna verzija: brute-force, k-means, IVF
│  ├─ omp/            OpenMP verzija (još prazno)
│  └─ main.cpp        komande gt-check, kmeans-check, ivf-check, ivf-search
├─ tests/test_seq.cpp testovi graničnih slučajeva (program ann_tests)
├─ scripts/           Python skripte: FAISS referenca, grafikoni (još prazno)
├─ results/           CSV rezultati i grafikoni (još prazno)
├─ data/              SIFT podaci (nije na GitHubu)
├─ build/             kompajlirani programi (nije na GitHubu)
└─ .venv/             Python okruženje (nije na GitHubu)
```

Fajlovi, redom kojim podaci prolaze kroz program:

| Fajl | Uloga |
|---|---|
| `include/simd_support.hpp` | `AlignedFloats`: niz floatova poravnat na 32 B (za AVX2). Windows `_aligned_malloc`, ostali `posix_memalign`. |
| `include/dataset.hpp`, `src/common/dataset.cpp` | `read_fvecs` / `read_ivecs` → `FloatMatrix` / `IntMatrix` (ravni nizovi, red i = `row(i)`). Provjerava dimenziju svakog zapisa i veličinu fajla. |
| `include/distance.hpp`, `src/distance/l2_scalar.cpp` | Kvadrat L2 udaljenosti i tip `DistFn`. Sav ostali kod prima udaljenost kao parametar, pa SIMD verzije ne mijenjaju IVF kod. Skalarna verzija se kompajlira bez vektorizacije (referenca). |
| `include/topk.hpp` | `TopK`: max-heap veličine k (vrh = najlošiji od k najboljih). Pri jednakoj udaljenosti pobjeđuje manji ID. `SearchResult` = format rezultata. |
| `include/bruteforce.hpp`, `src/seq/bruteforce_seq.cpp` | Tačan kNN – etalon za provjeru. |
| `include/kmeans.hpp`, `src/seq/kmeans_seq.cpp` | K-means (Lloyd): SplitMix64 + Fisher-Yates inicijalizacija, 25 iteracija, sume u double, prazni klasteri → najdalja tačka. |
| `include/ivf_index.hpp`, `src/seq/ivf_seq.cpp` | **Srce projekta.** `IvfLists` (CSR: `centroids`, `offset`, `list_data`, `list_ids`) i `IvfFlatSeq` sa `train`, `set_centroids`, `add`, `search`; `select_probes` bira nprobe listi. |
| `include/eval.hpp`, `src/common/eval.cpp` | `recall_at_k`, poređenje sa groundtruth-om po udaljenostima, provjera strukture listi. |
| `include/bench.hpp`, `src/common/bench.cpp` | Mjerni okvir iz Lab 1 (`perf::measure`: zagrijavanje, ponavljanja, medijan) – za fazu mjerenja. |

Raspored invertovanih listi (CSR):
```
centroids  [nlist × d]   centroidi iz k-meansa
offset     [nlist + 1]   lista c zauzima pozicije offset[c] .. offset[c+1]-1; offset[nlist] = N
list_data  [N × d]       svi vektori baze, poredani lista po lista (jedan kontinuiran niz)
list_ids   [N]           originalni ID vektora na svakoj poziciji
```

Kako IVF radi:
1. **train** – k-means na `learn` skupu dijeli prostor na `nlist` ćelija (centroida).
2. **add** – svaki vektor baze ide u listu najbližeg centroida (counting sort: dodjela,
   brojanje + prefix sum, kopiranje rastućim ID-em).
3. **search** – upit se poredi sa svim centroidima, bira se `nprobe` najbližih
   (`std::partial_sort`) i skeniraju se samo te liste kroz top-k heap.

Testovi (`ann_tests`) pokrivaju slučajeve koje SIFT ne pokriva: prazne klastere (duplikati tačaka),
jednake udaljenosti u top-k, da provjera listi **otkriva** pokvarene liste (offset, duplikat ID-a,
izmijenjen vektor, vektor u pogrešnoj listi) i da je pretraga sa `nprobe == nlist` identična
brute-force-u.

---

## 9. Python (faza mjerenja)

Za FAISS poređenje i grafikone (iz `implementacija/`). FAISS se **ne** linkuje u C++ projekat.

Windows:
```powershell
winget install Python.Python.3.12
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy matplotlib faiss-cpu
```
Python pokretati direktno kao `.venv\Scripts\python.exe skripta.py` (aktivacija okruženja u
PowerShellu može biti blokirana execution policy-jem).

macOS / Linux:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib faiss-cpu
```

---

## 10. IDE

- **VS Code:** ekstenzija *CMake Tools*, otvoriti folder `implementacija/`, izabrati kit
  „GCC … ucrt64" (Windows) ili Clang (macOS).
- **CLion:** na Windowsu u *Settings → Build → Toolchains* dodati MinGW sa putanjom `C:\msys64\ucrt64`.

---

## 11. Česti problemi

| Problem | Uzrok i rješenje |
|---|---|
| Program se pokrene, ništa ne ispiše (Windows), `$LASTEXITCODE` je `-1073741511` | Učitan tuđi `libstdc++-6.dll` (npr. iz QEMU-a ili Git-a koji su u PATH-u ispred MSYS2). Riješeno statičkim linkovanjem u `CMakeLists.txt`; ako se ipak desi – obrisati `build/` i ponovo konfigurisati. |
| `MSVC nije podržan` | CMake je izabrao Visual Studio. Koristiti `-G Ninja` i provjeriti da `g++ --version` radi u tom terminalu. |
| `g++` / `cmake` nije pronađen | `C:\msys64\ucrt64\bin` nije u PATH-u ili terminal nije ponovo otvoren. |
| `ne mogu otvoriti fajl ...siftsmall_base.fvecs` | Podaci nisu u `data/siftsmall/` ili su raspakovani u dodatni podfolder. |
| `pacman` javlja konflikte paketa | Pokrenuto `pacman -S` na neažuriranu bazu. Prvo `pacman -Syu` (više puta), pa instalacija. |
| macOS: greška za `-mcpu=native` | Stariji Apple clang. Ažurirati Xcode Command Line Tools. |
| Vremena mnogo duža nego kod drugih | Laptop na bateriji, ili je pokrenut Debug build. Mjeriti samo Release, na punjaču. |
| Brojevi (recall) drugačiji nego u ovom dokumentu | Ne bi smjelo da se desi – kod je deterministički. Javiti timu (kompajler, sistem, ispis). |
