# Tabele (generisano)

Generisano skriptom `python implementacija/scripts/vtune/tabela.py` iz CSV-ova u ovom folderu. Vremena su iz samostalnih mjerenja (bez VTune-a); ostale tabele iz finalnog VTune snimanja (izvjestaji po funkcijama sa `-inline-mode off`).

### Vrijeme pretrage (vremena-sift1m.txt)

| Varijanta | medijan (s) | QPS | ns po udaljenosti | GB/s | Recall@10 | ubrzanje |
|---|---|---|---|---|---|---|
| scalar | 11,735 | 85,2 | 89,6 | 5,7 | 0,9975 | 1,00× |
| scalar4 | 8,347 | 119,8 | 63,7 | 8,0 | 0,9975 | 1,41× |
| avx2 | 5,106 | 195,8 | 39,0 | 13,1 | 0,9975 | 2,30× |
| avx2x4 | 4,916 | 203,4 | 37,5 | 13,6 | 0,9975 | 2,39× |

### Vrijeme pretrage (vremena-sift10k-u-kesu.txt)

| Varijanta | medijan (s) | QPS | ns po udaljenosti | GB/s | Recall@10 | ubrzanje |
|---|---|---|---|---|---|---|
| scalar | 0,045 | 2228,3 | 87,1 | 5,8 | 1,0000 | 1,00× |
| scalar4 | 0,026 | 3801,9 | 51,0 | 9,9 | 1,0000 | 1,71× |
| avx2 | 0,009 | 11298,9 | 17,2 | 29,4 | 1,0000 | 5,07× |
| avx2x4 | 0,007 | 13505,4 | 14,4 | 35,2 | 1,0000 | 6,06× |

### Udio CPU vremena (hotspots)

| Varijanta | funkcija udaljenosti % | IvfFlatSeq::search % |
|---|---|---|
| scalar | 84,3 | 1,9 |
| scalar4 | 90,9 | 3,2 |
| avx2 | 86,7 | 4,1 |
| avx2x4 | 87,1 | 5,1 |

### Top-down metrike funkcije udaljenosti (uarch-exploration, P-jezgro)

| Metrika | scalar | scalar4 | avx2 | avx2x4 |
|---|---|---|---|---|
| CPI | 0,349 | 0,330 | 0,686 | 0,810 |
| Retiring % | 63,1 | 64,9 | 29,4 | 25,1 |
| — FP Scalar % | 35,8 | 56,4 | 0,6 | 0,7 |
| — FP Vector % | 0,0 | 0,0 | 28,0 | 35,5 |
| Front-End Bound % | 24,4 | 6,2 | 5,1 | 4,8 |
| Bad Speculation % | 4,3 | 0,2 | 0,9 | 3,4 |
| Back-End Bound % | 8,2 | 28,8 | 64,5 | 66,7 |
| — Memory Bound % | 5,5 | 16,8 | 35,5 | 41,2 |
| —— L1 Bound % | 4,1 | 5,8 | 0,0 | 0,0 |
| —— L3 Bound % | 2,7 | 7,3 | 7,3 | 5,2 |
| —— DRAM Bound % | 3,5 | 10,3 | 37,7 | 45,7 |
| — Core Bound % | 2,7 | 12,0 | 29,1 | 25,6 |
| Vector Capacity Usage % | 6,2 | 6,2 | 48,1 | 48,2 |

### Takti i instrukcije po elementu

| Metrika | scalar | scalar4 | avx2 | avx2x4 |
|---|---|---|---|---|
| taktova po elementu | 2,49 | 1,73 | 0,81 | 0,72 |
| instrukcija po elementu | 7,12 | 5,25 | 1,17 | 0,88 |

### Memorija (memory-access)

| Metrika | scalar | scalar4 | avx2 | avx2x4 |
|---|---|---|---|---|
| Memory Bound funkcije % | 3,7 | 13,2 | 35,3 | 40,6 |
| DRAM Bound funkcije % | 2,8 | 8,8 | 38,4 | 41,2 |
| DRAM GB/s, prosjek (sistem) | 20,9 | 21,5 | 21,8 | 21,1 |
| DRAM GB/s, izmjereni max (sistem) | 34,8 | 34,0 | 30,5 | 27,7 |
| DRAM GB/s, max platforme | 42,0 | 43,0 | 44,0 | 45,0 |

