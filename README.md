# Paralelni ANN indeks (IVF)

Projekat iz predmeta **Paralelni računarski sistemi**: ANN indeks zasnovan na invertovanim listama (IVF-Flat), implementiran sekvencijalno, a zatim paralelizovan na CPU-u pomoću OpenMP-a i SIMD-a (AVX2).

## Dokumentacija

U folderu `docs/` nalaze se folderi sa naučno-istraživačkim radovima, linkovima i ostalim dokumentima projekta. Organizacija prati zajednički folder na Google Driveu. Puni navodi, linkovi i kratak opis onoga što iz kojeg rada uzimamo nalaze se u [`docs/literatura.md`](docs/literatura.md).

- **`razliciti-pristupi-ann/`**: radovi o drugim porodicama ANN metoda, koji služe za pregled oblasti i obrazloženje zašto ih nismo izabrali: LSH (hashiranje), FLANN (stabla), HNSW (grafovi) i eksperimentalno poređenje ANN metoda (Li et al.).
- **`zasto-ivf/`**: radovi koji opisuju IVF i podupiru njegov izbor: Video Google (porijeklo ideje), Product Quantization for Nearest Neighbor Search (osnova algoritma) i Revisiting the Inverted Indices. The Inverted Multi-Index nije u repou kao PDF, nego samo kao link u `literatura.md`.
- **`paralelizacije/`**: radovi o paralelnoj implementaciji ANN pretrage: The Faiss Library, Quicker ADC (SIMD), te Billion-Scale Similarity Search with GPUs i Efficient Large-scale ANN Search on the GPU kao šira referenca.
- **`linkovi-sekvencijalne-implementacije.md`**: linkovi korisni za razumijevanje sekvencijalne implementacije (pseudokod i objašnjenje IVFFlat indeksa, Python tutorijal, FAISS repozitorij).

## Inicijalni plan implementacije

Implementacija je u **C++20** sa **CMake**-om. Mjerni okvir, poravnata memorija i obrazac SIMD verzija preuzimaju se iz laboratorijske vježbe 1. Kompajler je GCC ili Clang (Linux, WSL ili MSYS2): MSVC sa `/openmp` podržava samo OpenMP 2.0, pa novije klauzule (npr. `omp simd reduction`, taskovi, `collapse`) ne rade ili rade samo djelimično. Podaci su skupovi SIFT10K (razvoj) i SIFT1M (konačna mjerenja) sa [corpus-texmex.irisa.fr](http://corpus-texmex.irisa.fr/). SIFT1M potiče iz rada Jégou et al.: 100.000 vektora za učenje, 1.000.000 u bazi i 10.000 upita.

### 1. Sekvencijalna verzija
Algoritam prati IVF dio rada Jégou et al. (bez PQ kompresije).
- Učitavanje vektora iz `.fvecs` / `.ivecs` fajlova u poravnat ravan niz (vektor od 128 float-ova zauzima 512 B, pa je svaki vektor poravnat)
- Brute-force kNN kao referenca za tačnost
- K-means trening grubog kvantizatora (`nlist` centroida, fiksan seed)
- Invertovane liste u CSR rasporedu: vektori sortirani po listama, njihovi ID-evi i niz offseta
- Pretraga: `nprobe` najbližih centroida, skeniranje njihovih listi, top-k pomoću heapa
- Mjerenje: recall@10, vrijeme izgradnje, broj upita u sekundi (QPS)

### 2. SIMD
- L2 udaljenost u četiri verzije: skalarna, auto-vektorizovana, `omp simd` i AVX2 + FMA
- AVX2 verzija sa 1, 2 i 4 akumulatora: više nezavisnih zbirova bolje koristi procesor, ali samo dok podaci stižu iz keša
- FMA traži zastavicu `-mfma` i provjeru podrške u runtimeu, pored provjere za AVX2
- Izvještaj kompajlera o vektorizaciji (`-fopt-info-vec` za GCC, `-Rpass=loop-vectorize` za Clang) kao dokaz da su auto i `omp simd` verzije zaista vektorizovane. Bez `-ffast-math` kompajler ne smije promijeniti redoslijed float sabiranja, pa se auto verzija L2 udaljenosti vjerovatno neće vektorizovati.
- Ista IVF pretraga pokrenuta sa svakom verzijom udaljenosti

### 3. OpenMP
- K-means: paralelna dodjela tačaka centroidima, ažuriranje centroida preko lokalnih suma po niti. Sume svake niti poravnate su na 64 B (keš linija) da se izbjegne lažno dijeljenje, uz mjerenje sa i bez poravnanja.
- Izgradnja listi: paralelno brojanje i punjenje. Prefix sum nad `nlist` brojača ostaje sekvencijalan, jer je posao premali da bi se paralelizacija isplatila.
- Pretraga: paralelno po upitima, kao u FAISS-u (batch upita se dijeli na niti, pojedinačan upit ne). Poređenje statičkog i dinamičkog raspoređivanja: liste su nejednake dužine, pa upiti imaju različitu cijenu.
- Kombinacija OpenMP + AVX2

### 4. Mjerenja
- `nlist` ∈ {256, 1024, 4096} za SIFT1M, manje vrijednosti za SIFT10K
- `nprobe` ∈ {1, 8, 32, 128}
- Broj niti ∈ {1, 2, 4, 8}, plus broj logičkih jezgara (SMT)
- Ubrzanje i efikasnost; jako skaliranje (fiksan broj upita) i slabo skaliranje (broj upita raste sa brojem niti); procjena serijskog dijela po Amdahlovom zakonu
- Krive recall@10 naspram QPS-a umjesto pojedinačnih tačaka (metodologija kao kod Li et al.)
- Propusnost skeniranja (GB/s pročitanih vektora iz listi), poređena sa izmjerenom propusnošću memorije istog računara: pokazuje kada je pretraga ograničena memorijom
- Imbalance faktor listi, kao u FAISS-u: `nlist · Σ|Lᵢ|² / N²` (1 znači da su sve liste iste dužine)
- Poređenje svih verzija sa sekvencijalnom i sa FAISS `IndexIVFFlat` (CPU), uz isti `nlist`, `nprobe` i broj niti

### 5. Dodatni eksperimenti (ako ostane vremena)
- Raspored podataka: CSR sa kopiranim vektorima naspram listi koje čuvaju samo ID-eve, dok vektori ostaju u originalnom redoslijedu. Mjerenjem pokazati zašto je izabran CSR.
- Redoslijed petlji u pretrazi: upit po upit naspram lista po lista, gdje se svaka lista skenira jednom za cijeli blok upita koji je probaju, uz više veličina bloka. Vektor učitan iz memorije tada se koristi više puta.
- Transponovani blokovi od 8 vektora u listi: svaka AVX2 traka računa udaljenost do drugog vektora, bez horizontalnog sabiranja na kraju. Kandidati za heap filtriraju se maskom (`_mm256_cmp_ps`) umjesto grananjem.

## Tok implementacije

1. **Podaci i referenca.** Prvo se pišu čitač `.fvecs` fajlova i brute-force pretraga. Brute-force rezultat mora se poklopiti sa priloženim tačnim odgovorima (`groundtruth.ivecs`). Time su potvrđeni i podaci i računanje recalla prije nego što indeks postoji.
2. **K-means.** Trenira se na `learn` skupu. Provjera: ukupna udaljenost tačaka do centroida opada iz iteracije u iteraciju, a nijedan klaster ne ostaje prazan.
3. **Invertovane liste.** Svaki bazni vektor dodjeljuje se najbližem centroidu, broje se vektori po listi, prefix sum daje offsete, a vektori se kopiraju u jedan ravan niz. Provjera: zbir dužina listi jednak je broju vektora.
4. **Pretraga.** Za upit se nađe `nprobe` najbližih centroida, skeniraju se njihove liste i čuva top-k. Provjera: kad je `nprobe = nlist`, recall mora biti jednak brute-forceu (1,0). Ako malo odstupa, prvo se provjeravaju izjednačene udaljenosti, pa tek onda traži greška.
5. **Zaključavanje sekvencijalne verzije.** Kad su sve provjere prošle, sekvencijalna verzija se označava tagom u Gitu i postaje referenca za sve paralelne verzije.
6. **SIMD.** Mijenja se samo funkcija udaljenosti. Provjera: udaljenosti se porede sa skalarnom verzijom uz relativnu toleranciju, a recall mora ostati isti. Više akumulatora mijenja redoslijed sabiranja, pa se udaljenosti mogu razlikovati u posljednjim bitovima.
7. **OpenMP.** Paralelizuju se k-means, izgradnja listi i pretraga. Provjera: uz iste početne centroide recall mora biti isti kao u sekvencijalnoj verziji, uz toleranciju. Paralelna redukcija sabira drugim redoslijedom, pa se centroidi mogu razlikovati u posljednjim bitovima. Za ponovljiv rezultat sume se drže u `double`, uz statičko raspoređivanje i fiksan broj niti.
8. **Mjerenja i poređenje.** Sve verzije se mjere istim okvirom (zagrijavanje, više ponavljanja, medijan), a rezultat svake pretrage se koristi (npr. checksum ID-eva) da ga kompajler ne izbaci. Rezultati se porede sa FAISS-om.

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
| **Akumulator** | Promjenljiva ili SIMD registar u kojem se sabira djelimični zbir |
| **Efikasnost** | Ubrzanje podijeljeno brojem niti; 1 znači da su niti potpuno iskorištene |
| **Jako / slabo skaliranje** | Jako: posao je fiksan, a raste broj niti. Slabo: posao raste proporcionalno broju niti |
| **Amdahlov zakon** | Granica ubrzanja koju postavlja serijski dio programa |
| **Keš linija** | Najmanja jedinica prenosa između memorije i keša; 64 B na x86 procesorima |
| **Lažno dijeljenje** | Niti pišu u različite promjenljive koje leže u istoj keš liniji, pa se linija stalno seli između jezgara |
| **Propusnost memorije** | Broj bajtova u sekundi koje memorija može isporučiti procesoru (GB/s) |
| **SMT** | Simultaneous Multithreading: jedno fizičko jezgro izvršava više niti (logička jezgra) |
| **Imbalance faktor** | Mjera neravnomjernosti dužina listi; 1 znači da su sve liste iste dužine |
| **FAISS** | Biblioteka za vektorsku pretragu (Meta); koristi se kao spoljna referenca |
| **`IndexIVFFlat`** | FAISS implementacija IVF-Flat indeksa |
