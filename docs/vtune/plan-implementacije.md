# VTune analiza i optimizacija: plan implementacije

> **Za izvođača:** zadaci se rade redom, uz `- [ ]` praćenje. Zadaci 1, 2 i 4 ne traže VTune driver; zadatak 5 traži. Zadatak 3 se radi tek nakon zadatka 5, jer optimizacija slijedi iz nalaza.

**Cilj:** izmjeriti pretragu VTune-om, optimizovati nađeno usko grlo i izmjeriti ponovo, na identičnom indeksu.

**Arhitektura:** indeks se jednom napravi skalarnom verzijom i sačuva (`ivf_io`). Komanda `ivf-bench` ga učitava i mjeri samo pretragu kroz `perf::measure`, uz izbor funkcije udaljenosti po imenu. VTune se pokreće skriptom koja proces veže za P-jezgro i izvozi CSV izvještaje u `docs/vtune/rezultati/`.

**Tehnologije:** C++20, MSYS2 UCRT64 GCC, AVX2/FMA intrinsici, Intel VTune Profiler 2026.6 (CLI), PowerShell.

**Dizajn:** `docs/vtune/dizajn.md`

## Globalna ograničenja

- Kompajler GCC (MSYS2 UCRT64) ili Clang; MSVC nije podržan.
- Release zastavice ostaju: `-O3 -DNDEBUG -march=native -ffp-contract=off`; `l2_scalar.cpp` sa `-fno-tree-vectorize`.
- Rezultat pretrage (ID-evi i udaljenosti) sa bilo kojom varijantom udaljenosti mora biti **identičan** skalarnoj referenci na SIFT-u.
- Ispis programa bez naših slova (kao postojeći: `tacan`, `skenirano`); komentari na bosanskom.
- Podaci i fajlovi indeksa idu u `data/` (u `.gitignore`); sirovi VTune rezultati u `build/` (u `.gitignore`).
- Commit poruke bez ikakve atribucije alata ili AI-ja.

## Fokus pregleda

- Fajl indeksa napravljen za drugi `nlist`, seed ili skup → odbiti sa jasnom porukom, ne koristiti pogrešan indeks (test u zadatku 1).
- Skraćen ili pokvaren fajl indeksa → `std::runtime_error`, ne pad programa (test u zadatku 1).
- Nepoznato ime udaljenosti u `ivf-bench` → poruka sa spiskom ispravnih imena, izlazni kod 2 (provjera u zadatku 2).
- Procesor bez AVX2/FMA, a izabrana AVX2 varijanta → jasna greška umjesto "illegal instruction" (provjera u zadatku 3).
- Dimenzija koja nije višekratnik 8 ili 32 → AVX2 varijante tačne i na ostatku (test u zadatku 3, d ∈ {1, 7, 8, 31, 32, 33, 130}).

---

### Zadatak 1: Čuvanje i učitavanje indeksa

**Fajlovi:**
- Novo: `implementacija/include/ivf_io.hpp`, `implementacija/src/common/ivf_io.cpp`
- Izmjena: `implementacija/include/ivf_index.hpp` (metod `set_lists`), `implementacija/src/seq/ivf_seq.cpp`, `implementacija/CMakeLists.txt` (dodati `ivf_io.cpp` u `ann_core`)
- Test: `implementacija/tests/test_seq.cpp`

**Interfejsi:**
- Proizvodi: `void ann::save_ivf_lists(const IvfLists& lists, const std::filesystem::path& path);`
- Proizvodi: `IvfLists ann::load_ivf_lists(const std::filesystem::path& path);` (baca `std::runtime_error`)
- Proizvodi: `void IvfFlatSeq::set_lists(IvfLists lists);` (baca `std::invalid_argument` ako `offset.size() != nlist + 1` ili `offset.back() != n`)

Format fajla (little-endian): magični niz `PRSIVF01` (8 B), `uint64 nlist, d, n`, centroidi `nlist·d` float, `offset` `(nlist+1)` × `uint64`, `list_data` `n·d` float, `list_ids` `n` × `int32`. Učitavanje provjerava magični niz i da je veličina fajla tačno ona koju zaglavlje opisuje.

- [ ] **Korak 1: test `test_ivf_io()`** — mali indeks kao u `test_ivf_search`; sačuvati u privremeni fajl, učitati, pa `expect`: centroidi, `offset`, `list_data` i `list_ids` bit-identični; `search` sa učitanim indeksom identičan originalu; fajl sa izmijenjenim magičnim nizom baca izuzetak; fajl skraćen za 4 B baca izuzetak.
- [ ] **Korak 2:** build i `ann_tests` → test pada (funkcije ne postoje).
- [ ] **Korak 3:** implementirati `save_ivf_lists`, `load_ivf_lists`, `set_lists`.
- [ ] **Korak 4:** `ann_tests` → `SVI TESTOVI PROSLI (0 gresaka)`.
- [ ] **Korak 5:** commit.

### Zadatak 2: Komanda `ivf-bench`

**Fajlovi:**
- Izmjena: `implementacija/src/main.cpp`
- Novo: `implementacija/src/distance/registry.cpp`; izmjena `implementacija/include/distance.hpp`

**Interfejsi:**
- Proizvodi: `struct ann::DistanceVariant { const char* name; DistFn fn; bool needs_avx2_fma; };`
- Proizvodi: `const std::vector<DistanceVariant>& ann::distance_variants();` (za sada samo `{"scalar", l2_scalar, false}`)
- Proizvodi: `const DistanceVariant* ann::find_distance(std::string_view name);` (nullptr ako ne postoji)
- Proizvodi: `bool ann::cpu_has_avx2_fma();` (`__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")`; `registry.cpp` se kompajlira bez `-mavx2`)
- Komanda: `prs_ann ivf-bench <folder> <nlist> <nprobe> [broj_upita=1000] [ponavljanja=5] [udaljenost=scalar]`

Ponašanje: fajl indeksa `<folder>/<ime>_ivf_nlist<nlist>_seed20261107.bin`; ako ne postoji, napravi ga skalarnom verzijom (train + add), ispiše vrijeme i sačuva; ako postoji, učita ga i provjeri da `d` i `n` odgovaraju bazi. Za varijantu različitu od `scalar` prvo jednom izračuna skalarnu referencu. Mjerenje: `perf::measure` sa 1 zagrijavanjem i `ponavljanja` mjerenja. Ispis: varijanta, `nprobe`, upiti, medijan i minimum (s), QPS, skenirano po upitu, ns po udaljenosti (medijan / ukupno skeniranih), GB/s skeniranja (skenirano · d · 4 B / medijan), Recall@10, identično skalarnoj referenci `x / nq`. Na kraju jedan red `CSV: ...` sa istim vrijednostima.

- [ ] **Korak 1:** implementirati registry i `run_ivf_bench`.
- [ ] **Korak 2:** `prs_ann ivf-bench data/siftsmall 64 8 0 5 scalar` → Recall@10 `0.9770`, poruka da je indeks napravljen i sačuvan.
- [ ] **Korak 3:** isto ponovo → poruka da je indeks učitan, isti Recall@10.
- [ ] **Korak 4:** `prs_ann ivf-bench data/siftsmall 64 8 0 5 nepostoji` → spisak imena, izlazni kod 2.
- [ ] **Korak 5:** `ann_tests` prolazi; commit.

### Zadatak 3: Optimizovane varijante udaljenosti (nakon zadatka 5)

Radi se samo ako zadatak 5 potvrdi da je usko grlo zavisnost u `l2_scalar`. Inače se ovaj zadatak prepisuje prema nalazu.

**Fajlovi:**
- Novo: `implementacija/src/distance/l2_unrolled.cpp` (zastavica `-fno-tree-vectorize`, kao skalarna), `implementacija/src/distance/l2_avx2.cpp` (zastavice `-mavx2 -mfma`)
- Izmjena: `implementacija/include/distance.hpp`, `implementacija/src/distance/registry.cpp`, `implementacija/CMakeLists.txt`
- Test: `implementacija/tests/test_seq.cpp`

**Interfejsi:**
- Proizvodi: `float ann::l2_scalar4(const float*, const float*, std::size_t);` — skalarna, 4 nezavisna akumulatora, ostatak skalarno
- Proizvodi: `float ann::l2_avx2_fma1(...)` — AVX2 + FMA, 1 akumulator (8 po koraku), ostatak skalarno
- Proizvodi: `float ann::l2_avx2_fma4(...)` — AVX2 + FMA, 4 akumulatora (32 po koraku), pa koraci po 8, pa skalarni ostatak
- Registry: `scalar`, `scalar4`, `avx2`, `avx2x4`; zadnja dva sa `needs_avx2_fma = true`. `ivf-bench` prije pokretanja provjeri `needs_avx2_fma && !cpu_has_avx2_fma()` i tada ispiše grešku (izlazni kod 2) umjesto da program padne na nepodržanoj instrukciji.

- [ ] **Korak 1: test `test_distance_variants()`** — za d ∈ {1, 7, 8, 31, 32, 33, 128, 130}: vektori sa cijelim vrijednostima 0..255 → svaka varijanta `==` `l2_scalar` (tačno, zbir < 2²⁴); vektori sa necijelim vrijednostima → relativna greška < 1e-5. AVX2 varijante se preskaču uz ispis ako `cpu_has_avx2_fma()` nije tačno.
- [ ] **Korak 2:** build → test pada (funkcije ne postoje).
- [ ] **Korak 3:** implementirati tri funkcije i dodati ih u registry.
- [ ] **Korak 4:** `ann_tests` prolazi; `ivf-bench data/siftsmall 64 8 0 5 avx2x4` → identično skalarnoj `100 / 100`, Recall@10 `0.9770`.
- [ ] **Korak 5:** commit.

### Zadatak 4: VTune skripta

**Fajlovi:**
- Novo: `implementacija/scripts/vtune/collect.ps1`
- Izmjena: `implementacija/CMakeLists.txt` (`CMAKE_CXX_FLAGS_RELWITHDEBINFO` = `-O3 -DNDEBUG -g`, sa istim `-march=native` kao Release)

**Interfejsi:**
- `collect.ps1 -Exe <putanja> -Label <ime> -Dataset <folder> -Nlist <n> -Nprobe <p> -Queries <q> -Repeats <r> -Distance <ime> [-Analyses hotspots,uarch-exploration,memory-access] [-Cpu 2]`
- Za svaku analizu: `vtune -collect <analiza>` (hotspots sa `-knob sampling-mode=hw`) nad `cmd /c start "" /b /wait /affinity <maska> <exe> ivf-bench ...`, rezultat u `build/vtune-results/<label>-<analiza>`; zatim `vtune -report summary` i `vtune -report hotspots -group-by function` u CSV, u `docs/vtune/rezultati/<label>/<analiza>-summary.csv` i `<analiza>-functions.csv`.

- [ ] **Korak 1:** napisati skriptu.
- [ ] **Korak 2:** probni rad na `data/siftsmall` (nakon instalacije drivera) → CSV fajlovi postoje i sadrže `l2_scalar` među funkcijama.
- [ ] **Korak 3:** commit.

### Zadatak 5: Početno mjerenje i analiza

- [ ] **Korak 1:** napraviti indeks za SIFT1M: `ivf-bench data/sift 256 32 1000 3 scalar` (pravi i čuva indeks; liste ~516 MB, `nprobe` 32 skenira ~13 % baze ≈ 64 MB po upitu, više od L3 keša od 18 MB).
- [ ] **Korak 2:** vremena za `scalar` (5 ponavljanja, vezano za P-jezgro).
- [ ] **Korak 3:** `collect.ps1 -Label scalar` sa sve tri analize.
- [ ] **Korak 4:** iz CSV-a izdvojiti: CPI, Retiring, Front-End Bound, Bad Speculation, Back-End Bound, Memory Bound (L1/L2/L3/DRAM), Core Bound, propusnost memorije; zapisati zaključak (potvrđena ili odbačena hipoteza).
- [ ] **Korak 5:** commit CSV-ova i bilješki.

### Zadatak 6: Mjerenje nakon optimizacije

- [ ] **Korak 1:** vremena za `scalar`, `scalar4`, `avx2`, `avx2x4` na istom indeksu, sa istim parametrima; svaka varijanta `identicno ... 1000 / 1000`.
- [ ] **Korak 2:** `collect.ps1 -Label avx2x4` (i `scalar4`, ako zadatak 5 pokaže da je bitan) sa sve tri analize.
- [ ] **Korak 3:** tabela prije/poslije istih metrika; commit.

### Zadatak 7: Izvještaj

**Fajlovi:**
- Novo: `docs/vtune/izvjestaj.md`
- Izmjena: `implementacija/README.md` (komanda `ivf-bench`, RelWithDebInfo build, kako pokrenuti VTune skriptu)

- [ ] **Korak 1:** izvještaj: postavka (računar, skup, parametri, vezivanje za P-jezgro), metode (tri analize i šta mjere), metrike prije, hipoteza i potvrda, optimizacija i eksperiment u četiri varijante, metrike poslije, novo usko grlo, zaključak u nekoliko rečenica za prezentaciju.
- [ ] **Korak 2:** README dopuna.
- [ ] **Korak 3:** commit.
