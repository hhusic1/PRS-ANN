# Paralelni ANN indeks (IVF)

Projekat iz predmeta **Paralelni računarski sistemi**: ANN indeks zasnovan na invertovanim listama (IVF-Flat), implementiran sekvencijalno, a zatim paralelizovan na CPU-u pomoću OpenMP-a i SIMD-a (AVX2).

## Dokumentacija

U folderu `docs/` nalaze se folderi sa naučno-istraživačkim radovima, linkovima i ostalim dokumentima projekta. Organizacija prati zajednički folder na Google Driveu:

- **`razliciti-pristupi-ann/`**: radovi o drugim porodicama ANN metoda, koji služe za pregled oblasti i obrazloženje zašto ih nismo izabrali: LSH (hashiranje), FLANN (stabla), HNSW (grafovi) i eksperimentalno poređenje ANN metoda (Li et al.).
- **`zasto-ivf/`**: radovi koji opisuju IVF i podupiru njegov izbor: Video Google (porijeklo ideje), Product Quantization for Nearest Neighbor Search (osnova algoritma), The Inverted Multi-Index i Revisiting the Inverted Indices.
- **`paralelizacije/`**: radovi o paralelnoj implementaciji ANN pretrage: The Faiss Library, Quicker ADC (SIMD), te Billion-Scale Similarity Search with GPUs i Efficient Large-scale ANN Search on the GPU kao šira referenca.
- **`linkovi-sekvencijalne-implementacije.md`**: linkovi korisni za razumijevanje sekvencijalne implementacije (pseudokod i objašnjenje IVFFlat indeksa, Python tutorijal, FAISS repozitorij).

## Inicijalni plan implementacije

Implementacija je u **C++20** sa **CMake**-om. Mjerni okvir, poravnata memorija i obrazac SIMD verzija preuzimaju se iz laboratorijske vježbe 1. Podaci su skupovi SIFT10K (razvoj) i SIFT1M (konačna mjerenja) sa [corpus-texmex.irisa.fr](http://corpus-texmex.irisa.fr/).

### 1. Sekvencijalna verzija
- Učitavanje vektora iz `.fvecs` / `.ivecs` fajlova
- Brute-force kNN kao referenca za tačnost
- K-means trening grubog kvantizatora (`nlist` centroida, fiksan seed)
- Invertovane liste u CSR rasporedu: vektori sortirani po listama, njihovi ID-evi i niz offseta
- Pretraga: `nprobe` najbližih centroida, skeniranje njihovih listi, top-k pomoću heapa
- Mjerenje: recall@10, vrijeme izgradnje, broj upita u sekundi (QPS)

### 2. SIMD
- L2 udaljenost u četiri verzije: skalarna, auto-vektorizovana, `omp simd` i AVX2 + FMA
- Više akumulatora u AVX2 verziji radi boljeg iskorištenja procesora
- Ista IVF pretraga pokrenuta sa svakom verzijom udaljenosti

### 3. OpenMP
- K-means: paralelna dodjela tačaka centroidima, ažuriranje centroida preko lokalnih suma po niti
- Izgradnja listi: paralelno brojanje i punjenje
- Pretraga: paralelno po upitima, poređenje statičkog i dinamičkog raspoređivanja
- Kombinacija OpenMP + AVX2

### 4. Mjerenja
- `nprobe` ∈ {1, 8, 32, 128}
- Broj niti ∈ {1, 2, 4, 8}
- Poređenje svih verzija sa sekvencijalnom i sa FAISS `IndexIVFFlat` (CPU)

## Tok implementacije

1. **Podaci i referenca.** Prvo se pišu čitač `.fvecs` fajlova i brute-force pretraga. Brute-force rezultat mora se poklopiti sa priloženim tačnim odgovorima (`groundtruth.ivecs`). Time su potvrđeni i podaci i računanje recalla prije nego što indeks postoji.
2. **K-means.** Trenira se na `learn` skupu. Provjera: ukupna udaljenost tačaka do centroida opada iz iteracije u iteraciju, a nijedan klaster ne ostaje prazan.
3. **Invertovane liste.** Svaki bazni vektor dodjeljuje se najbližem centroidu, broje se vektori po listi, prefix sum daje offsete, a vektori se kopiraju u jedan ravan niz. Provjera: zbir dužina listi jednak je broju vektora.
4. **Pretraga.** Za upit se nađe `nprobe` najbližih centroida, skeniraju se njihove liste i čuva top-k. Provjera: kad je `nprobe = nlist`, recall mora biti jednak brute-forceu (1,0).
5. **Zaključavanje sekvencijalne verzije.** Kad su sve provjere prošle, sekvencijalna verzija se označava tagom u Gitu i postaje referenca za sve paralelne verzije.
6. **SIMD.** Mijenja se samo funkcija udaljenosti. Provjera: rezultati iste pretrage moraju dati isti recall kao skalarna verzija.
7. **OpenMP.** Paralelizuju se k-means, izgradnja listi i pretraga. Provjera: uz iste početne centroide, recall mora biti isti kao u sekvencijalnoj verziji.
8. **Mjerenja i poređenje.** Sve verzije se mjere istim okvirom (zagrijavanje, više ponavljanja, medijan), a rezultati se porede sa FAISS-om.

Svaki korak se razvija na SIFT10K, a kad radi ispravno, provjerava se i na SIFT1M.

## Pojmovi i skraćenice

| Pojam | Značenje |
| --- | --- |
| **IVF** | Inverted File Index, indeks sa invertovanim listama: prostor se k-meansom dijeli na ćelije, a svaki vektor se smješta u listu svoje ćelije |
| **IVF-Flat** | IVF koji u listama čuva originalne (nekomprimovane) vektore; verzija koju implementiramo |
| **IVFADC / PQ** | IVF sa Product Quantization kompresijom vektora (rad Jégou et al.); nije dio naše implementacije |
| **LSH, FLANN, HNSW** | Druge porodice ANN metoda: hashiranje, stabla i grafovi; opisane u literaturi |
| **SIFT10K / SIFT1M** | Standardni skupovi od 10.000 odnosno 1.000.000 vektora dimenzije 128 |
| **`.fvecs` / `.ivecs`** | Binarni formati: za svaki vektor dimenzija (4 bajta), pa vrijednosti kao float (`.fvecs`) ili int (`.ivecs`) |
| **k-means** | Algoritam klasterovanja koji naizmjenično dodjeljuje tačke najbližem centroidu i pomjera centroide na prosjek dodijeljenih tačaka |
| **Centroid** | Središte jedne ćelije (klastera) |
| **Grubi kvantizator** | Skup centroida koji određuje u koju listu ide svaki vektor |
| **`nlist`** | Broj centroida, odnosno broj invertovanih listi |
| **`nprobe`** | Broj najbližih listi koje se pretražuju za jedan upit; veći `nprobe` daje veći recall, ali sporiju pretragu |
| **`k` / top-k** | Broj traženih najbližih susjeda i postupak čuvanja k najboljih kandidata |
| **Heap** | Struktura podataka (prioritetni red) koja efikasno čuva k najboljih kandidata tokom pretrage |
| **Invertovana lista** | Spisak vektora koji pripadaju jednom centroidu |
| **CSR raspored** | Compressed Sparse Row: sve liste u jednom ravnom nizu, uz niz offseta koji kaže gdje svaka lista počinje |
| **Offset** | Pozicija početka liste u ravnom nizu |
| **Prefix sum (scan)** | Niz kumulativnih zbirova; iz broja vektora po listama daje offsete |
| **Recall@10** | Udio pravih 10 najbližih susjeda koje je pretraga pronašla (od 0 do 1) |
| **QPS** | Queries Per Second, broj upita obrađenih u sekundi |
| **OpenMP** | Standard za paralelizaciju u C/C++ pomoću direktiva `#pragma omp` |
| **SIMD** | Single Instruction, Multiple Data: jedna instrukcija obrađuje više brojeva odjednom |
| **FMA** | Fused Multiply-Add: množenje i sabiranje u jednoj instrukciji |
| **FAISS** | Biblioteka za vektorsku pretragu (Meta); koristi se kao spoljna referenca |
| **`IndexIVFFlat`** | FAISS implementacija IVF-Flat indeksa |