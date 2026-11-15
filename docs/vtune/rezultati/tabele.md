# Tabele (generisano)

Generisano skriptom `python implementacija/scripts/vtune/tabela.py`. Vremena su iz samostalnih mjerenja (bez VTune-a). VTune tabele za `scalar` i `avx2x4` su iz ponovljenog snimanja na neopterećenom računaru; izvještaji po funkcijama su sa `-inline-mode off`.

### Vrijeme pretrage (vremena-sift1m.txt)

| Varijanta | medijan (s) | QPS | ns po udaljenosti | GB/s | Recall@10 | ubrzanje |
|---|---|---|---|---|---|---|
| scalar | 11,735 | 85,2 | 89,6 | 5,7 | 0,9975 | 1,00× |
| scalar4 | 8,347 | 119,8 | 63,7 | 8,0 | 0,9975 | 1,41× |
| auto | 9,323 | 107,3 | 71,2 | 7,2 | 0,9975 | 1,26× |
| omp | 5,537 | 180,6 | 42,3 | 12,1 | 0,9975 | 2,12× |
| avx2 | 5,106 | 195,8 | 39,0 | 13,1 | 0,9975 | 2,30× |
| avx2x4 | 4,916 | 203,4 | 37,5 | 13,6 | 0,9975 | 2,39× |

### Vrijeme pretrage (vremena-sift10k-u-kesu.txt)

| Varijanta | medijan (s) | QPS | ns po udaljenosti | GB/s | Recall@10 | ubrzanje |
|---|---|---|---|---|---|---|
| scalar | 0,045 | 2228,3 | 87,1 | 5,8 | 1,0000 | 1,00× |
| scalar4 | 0,026 | 3801,9 | 51,0 | 9,9 | 1,0000 | 1,71× |
| auto | 0,030 | 3300,1 | 58,8 | 8,6 | 1,0000 | 1,48× |
| omp | 0,009 | 11227,4 | 17,3 | 29,3 | 1,0000 | 5,04× |
| avx2 | 0,009 | 11298,9 | 17,2 | 29,4 | 1,0000 | 5,07× |
| avx2x4 | 0,007 | 13505,4 | 14,4 | 35,2 | 1,0000 | 6,06× |

### Udio CPU vremena (hotspots)

| Varijanta | funkcija udaljenosti % | IvfFlatSeq::search % |
|---|---|---|
| scalar | 93,6 | 2,4 |
| avx2x4 | 86,4 | 5,1 |

### Top-down metrike funkcije udaljenosti (uarch-exploration, P-jezgro)

| Metrika | scalar | avx2x4 |
|---|---|---|
| CPI | 0,278 | 0,849 |
| Retiring % | 68,9 | 24,6 |
| — FP Scalar % | 36,4 | 0,7 |
| — FP Vector % | 0,0 | 37,9 |
| Front-End Bound % | 15,3 | 4,9 |
| Bad Speculation % | 6,0 | 1,3 |
| Back-End Bound % | 9,8 | 69,2 |
| — Memory Bound % | 6,1 | 41,9 |
| —— L1 Bound % | 1,4 | 0,1 |
| —— L3 Bound % | 2,6 | 7,1 |
| —— DRAM Bound % | 4,3 | 43,6 |
| — Core Bound % | 3,7 | 27,2 |
| Vector Capacity Usage % | 6,2 | 48,2 |

### Takti i instrukcije po elementu

| Metrika | scalar | avx2x4 |
|---|---|---|
| taktova po elementu | 1,98 | 0,75 |
| instrukcija po elementu | 7,12 | 0,89 |

### Memorija (memory-access)

| Metrika | scalar | avx2x4 |
|---|---|---|
| Memory Bound funkcije % | 5,6 | 42,2 |
| DRAM Bound funkcije % | 4,1 | 43,0 |
| DRAM GB/s, prosjek (sistem) | 15,0 | 22,1 |
| DRAM GB/s, izmjereni max (sistem) | 28,1 | 29,8 |
| DRAM GB/s, max platforme | 44,0 | 45,0 |

