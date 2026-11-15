# Nalaz 1: početno stanje (skalarna udaljenost)

Mjerenje: `ivf-bench data/sift 256 32 1000`, SIFT1M, `nlist` 256, `nprobe` 32 (13,08 % baze po upitu), indeks sačuvan i isti za sva mjerenja, proces vezan za P-jezgro (logičko jezgro 2), Intel Core i5-12500H. Sirovi CSV-ovi: `rezultati/scalar/`.

## Vrijeme (bez VTune-a, 5 ponavljanja, medijan)

| Metrika | Vrijednost |
|---|---|
| 1000 upita | 11,66 s |
| QPS | 85,7 |
| ns po udaljenosti (128 dimenzija) | 89,0 |
| propusnost skeniranja | 5,74 GB/s |
| Recall@10 | 0,9975 |

## VTune: microarchitecture exploration, funkcija `l2_scalar` (77 % CPU vremena)

| Metrika | Vrijednost | Značenje |
|---|---|---|
| CPI / IPC | 0,282 / 3,54 | procesor izvršava 3,5 instrukcije po taktu |
| Retiring | 69,3 % | većinu vremena se radi koristan posao |
| — FP Scalar | 35,6 % | sva FP aritmetika je skalarna |
| — FP Vector (128 i 256 bita) | 0,0 % | SIMD se ne koristi |
| Vector Capacity Usage (FPU) | 6,2 % | koristi se mali dio širine vektorskih jedinica |
| Front-End Bound | 17,2 % | mala petlja (Loop Stream Detector, 8,5 %) |
| Bad Speculation | 5,8 % | promašeno predviđanje izlaza iz petlje (jednom po udaljenosti) |
| Back-End Bound | 7,6 % | |
| — Memory Bound | 5,1 % | **ne čeka se na keš ni memoriju** |
| — DRAM Bound | 3,5 % | |
| — Core Bound | 2,5 % | |

Memory access analiza potvrđuje: Memory Bound 6,8 % za `l2_scalar` (DRAM 4,0 %), a program skenira 5,7 GB/s od 45 GB/s koliko platforma može.

## Takti i instrukcije po elementu

Obrađeno je 4 × 1000 × (130.769 + 256) × 128 = 67,08 milijardi elemenata (zagrijavanje + 3 mjerenja, uz udaljenosti do centroida).

| | Vrijednost |
|---|---|
| taktova po elementu | 134,74 · 10⁹ / 67,08 · 10⁹ = **2,01** |
| instrukcija po elementu | 477,44 · 10⁹ / 67,08 · 10⁹ = **7,12** |

Asembler petlje (GCC 13.2, `-O3 -march=native -fno-tree-vectorize -ffp-contract=off`):

```asm
loop:  vmovss  xmm0, [rcx+rax*4]          ; a[i]
       vsubss  xmm0, xmm0, [rdx+rax*4]    ; a[i] - b[i]
       add     rax, 1
       cmp     r8, rax
       vmulss  xmm0, xmm0, xmm0           ; razlika^2
       vaddss  xmm1, xmm1, xmm0           ; sum += ...   <- zavisi od prethodnog sum
       jne     loop
```

## Tumačenje

1. **Ne čekamo na memoriju.** Memory Bound je ~5 %, a propusnost 8× ispod mogućnosti platforme. Hardverski prefetcher stiže dovući liste, jer se čitaju sekvencijalno (CSR).
2. **Tempo određuje zavisnost podataka.** Petlja traje tačno 2 takta po elementu: koliko traje jedno FP sabiranje na Golden Cove P-jezgru. Svako `vaddss` čeka rezultat prethodnog, kroz isti registar `xmm1`. Za ta 2 takta procesor izvrši ostalih 6 instrukcija, pa VTune vidi zauzete portove (Retiring 69 %, 3+ porta zauzeta 76 % ciklusa) umjesto klasičnog "Core Bound". Lanac je skriven iza korisnog posla, ali i dalje postavlja gornju granicu brzine.
3. **Sav posao je skalaran.** FP Vector je 0 %: za svaki element izvrši se 7 instrukcija, iako AVX2 registar prima 8 floatova.

Hipoteza iz dizajna ("zavisnost u `l2_scalar`") je potvrđena kao granica brzine (2 takta po elementu). VTune top-down ju je, međutim, svrstao u Retiring, a ne u Core Bound, pa mjerenje dodaje i drugi uzrok: skalarne instrukcije (FP Vector 0 %).

## Predviđanja za optimizaciju (zapisana prije mjerenja)

| Varijanta | Šta se mijenja | Očekivana granica | Očekivano |
|---|---|---|---|
| `scalar4` | 4 nezavisna akumulatora, i dalje skalarno | broj instrukcija (~5 po elementu, IPC ~4–5) | ~1–1,2 takta po elementu, ~1,7–2× brže |
| `avx2` | AVX2 + FMA, 1 akumulator | lanac FMA: ~4 takta po koraku od 8 elemenata | ~0,5 takta po elementu; zavisnost postaje vidljiva kao Core Bound |
| `avx2x4` | AVX2 + FMA, 4 akumulatora | memorija: ~40 GB/s bi bilo potrebno, a jedno jezgro ne dobije toliko | ~4–5× brže od `scalar`, VTune pokazuje Memory / DRAM Bound |
