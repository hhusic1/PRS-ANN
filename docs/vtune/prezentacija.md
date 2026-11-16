# Vodič za prezentaciju: VTune analiza i SIMD faza

Pratilac uz [`izvjestaj.md`](izvjestaj.md), koji ima sve brojke i detalje. Ovdje je redoslijed izlaganja, šta se na kojem koraku pokazuje, šta koja brojka znači i odgovori na vjerovatna pitanja.

Trajanje: 10–12 minuta, plus pitanja.

## Priča u tri rečenice

1. Izmjerili smo gdje pretraga troši vrijeme. VTune je pokazao da **ne čekamo na memoriju**, nego na **lanac zavisnih sabiranja** i **skalarni kod**.
2. SIMD verzija (AVX2 + FMA, 4 akumulatora) je **2,39× brža** na SIFT1M i **6× brža** kad su podaci u kešu, uz identične rezultate.
3. Nakon toga usko grlo postaje **propusnost memorije jednog jezgra** (~13,6 GB/s), što je argument za sljedeću, OpenMP fazu.

Ako asistent upamti samo jedno, neka bude ovo: *mjerenje je pokazalo šta da optimizujemo, a mjerenje poslije pokazalo je da je problem riješen i šta je sljedeći problem.*

## Redoslijed izlaganja

| # | Korak | Šta pokazuješ | Ključna poruka | Vrijeme |
|---|---|---|---|---|
| 1 | Pitanje i postavka | izvještaj §1 (tabela postavke) | po upitu se skenira 64 MB, više od L3 keša (18 MB), pa vidimo stvarno ponašanje memorije | 1 min |
| 2 | Gdje je vrijeme | snimak 03 (Hotspots) | 93 % vremena je u jednoj funkciji, `l2_scalar`: nju optimizujemo | 1 min |
| 3 | Zašto je spora | snimak 04, grafikon 10 (gornja traka), asembler iz §3 | ne čekamo memoriju (Memory Bound 6 %); sav posao je skalaran (FP Vector 0 %); 2 takta po elementu = kašnjenje jednog sabiranja | 2 min |
| 4 | Potvrda | snimak 05 (Memory Access) | i druga analiza kaže isto: Memory Bound 7 %, DRAM 5 % | 0,5 min |
| 5 | Šta smo probali | izvještaj §4 (tabela varijanti) | šest varijanti, svaka mijenja jednu stvar, pa vidimo šta koliko donosi | 1,5 min |
| 6 | Rezultat | snimci 01 i 02, grafikon 09 | 2,39× iz RAM-a, 6,06× iz keša; rezultat identičan za svih 1000 upita | 1,5 min |
| 7 | VTune poslije | snimak 07 (06 i 08 po potrebi), grafikon 10 (donja traka) | SIMD radi (FP Vector 38 %); sada čekamo memoriju (Memory Bound 42 %, DRAM 44 %) | 1,5 min |
| 8 | Novo usko grlo | grafikon 11 | iz RAM-a jedno jezgro ne povuče više od ~13,6 GB/s, iz keša ista funkcija čita 35 GB/s | 1 min |
| 9 | Zaključak | izvještaj §9 | sljedeći korak: više jezgara (OpenMP), manje bajtova (`uint8`), grupisanje upita | 1 min |
| (po potrebi) | Režim napajanja | grafikon 12 | uz Energy Saver je sve ~1,3× sporije, ali ubrzanje i poredak ostaju isti | 0,5 min |

Snimci su u [`slike/`](slike/) i već su ugrađeni u izvještaj (§11), a grafikoni u §5–§8. Grafikon 12 pokazati ako asistent pita za uslove mjerenja, ili odmah poslije koraka 6. Najlakše je prezentovati direktno iz izvještaja na GitHubu: otvoriti PR, pa *Files changed*, pa `docs/vtune/izvjestaj.md`, pa *View file*.

### Šta reći na svakom koraku

**1. Postavka.** "Mjerimo samo pretragu: 1000 upita na milion SIFT vektora, `nprobe` 32. Po upitu se pregleda 13 % baze, oko 64 MB, a to je više nego što staje u keš. Proces je vezan za jedno P-jezgro, jer je E-jezgro 2,3× sporije. Indeks je napravljen jednom i sačuvan, pa sve varijante rade na istom indeksu. Mjereno je uz Windows Energy Saver; kontrolno mjerenje u režimu Best performance je u izvještaju i potvrđuje iste zaključke."

**2. Hotspots (snimak 03).** "Prvo pitanje je gdje je vrijeme. Odgovor je jasan: 93 % je u funkciji udaljenosti. Ostatak programa nije bitan za ubrzanje."

**3. Top-down (snimak 04 i gornja traka grafikona 10).** "VTune svaki takt procesora svrstava u kategorije. Kod nas je 69 % koristan posao (Retiring), a čekanje na memoriju samo 6 %. Dakle, ne čekamo na keš ni RAM. Ali FP Vector je 0 %: sve se računa jedan po jedan broj. U asembleru je 7 instrukcija po elementu, a izmjereno je tačno 2 takta po elementu. To je kašnjenje jednog FP sabiranja: svako sabiranje čeka prethodno, jer sve ide u istu promjenljivu `sum`. To je zavisnost podataka."

**4. Memory Access (snimak 05).** "Posebna memorijska analiza to potvrđuje: Memory Bound 7 %, DRAM 5 %."

**5. Varijante.** "Imali smo dvije hipoteze: zavisnost i skalarni kod. Zato smo napravili verzije koje ih odvajaju: `scalar4` uklanja samo zavisnost, `auto` i `omp` testiraju koliko može kompajler, a `avx2` i `avx2x4` su ručni SIMD sa jednim i sa četiri lanca."

**6. Rezultat (snimci 01 i 02, grafikon 09).** "Ručni AVX2 sa četiri akumulatora je 2,39× brži iz RAM-a i 6× brži iz keša. Program sam provjerava da je rezultat identičan skalarnom, ID po ID, za svih 1000 upita. Zanimljivo: kompajler sam daje samo 1,26×, jer bez `-ffast-math` ne smije promijeniti redoslijed sabiranja, pa lanac ostaje. `omp simd` mu to dozvoli i daje 2,12×."

**7. VTune poslije (snimak 07 i donja traka grafikona 10).** "Isti VTune pogled poslije optimizacije: FP Vector je sada 38 %, instrukcija po elementu je 8× manje. Ali Memory Bound je skočio sa 6 na 42 %, a unutar DRAM Bound-a 91 % otpada na propusnost (Memory Bandwidth). Procesor sada čeka podatke iz RAM-a."

**8. Zid memorije (grafikon 11).** "Plave trake su podaci iz RAM-a: sve brze varijante zapnu na 12–13,6 GB/s. Narandžaste su isti kod nad podacima u kešu: 35 GB/s. Računanje više nije problem: procesor bi mogao računati oko 10× brže nego što memorija dostavlja podatke."

**9. Zaključak.** "Početni problem bila je zavisnost i skalarni kod, a ne memorija. SIMD ga je riješio. Novi problem je memorija jednog jezgra, pa je sljedeći korak OpenMP: više jezgara znači više istovremenih zahtjeva prema memoriji."

## Šta znači koja brojka

### Brojke iz našeg programa (`ivf-bench`)

| Brojka | Značenje | Naše vrijednosti |
|---|---|---|
| medijan (s) | vrijeme 1000 upita; od 5 ponavljanja uzima se srednje po veličini, otporno na slučajne smetnje | 11,74 s → 4,92 s |
| QPS | upita u sekundi (1000 / vrijeme) | 85 → 203 |
| ubrzanje | vrijeme skalarne / vrijeme varijante | 2,39× (6,06× u kešu) |
| ns po udaljenosti | prosječno vrijeme jednog poređenja upit–vektor (128 brojeva) | 89,6 → 37,5 ns |
| GB/s | koliko bajtova vektora program pregleda u sekundi | 5,7 → 13,6 (u kešu 35,2) |
| Recall@10 | udio pravih 10 najbližih susjeda koje IVF nađe; mjera kvaliteta, ne brzine | 0,9975 u svim varijantama |
| identično x / 1000 | za koliko upita su ID-evi i udaljenosti tačno isti kao kod skalarne verzije | 1000 / 1000 |

### VTune: gdje je vrijeme

| Brojka | Značenje | Naše vrijednosti |
|---|---|---|
| Elapsed Time | ukupno trajanje snimanja (uključuje učitavanje indeksa od 516 MB) | na vrhu snimaka |
| CPU Time / udio funkcije | koliko je procesorskog vremena potrošeno u kojoj funkciji | `l2_scalar` 93 % |

### VTune: top-down (kako procesor troši taktove)

Procesor u svakom taktu ima nekoliko "mjesta" (pipeline slots) za instrukcije. Top-down kaže šta se desilo sa svakim mjestom:

| Kategorija | Značenje | scalar → avx2x4 | Kako čitati |
|---|---|---|---|
| Retiring | mjesto je iskorišteno za instrukciju koja je završena | 68,9 → 24,6 % | pad **nije loš**: isti posao se radi sa 8× manje instrukcija, ostatak vremena je čekanje |
| Front-End Bound | procesor ne stiže dohvatiti i dekodirati instrukcije | 15,3 → 4,9 % | manje instrukcija = manje posla za dekoder |
| Bad Speculation | posao bačen zbog pogrešno pogođenog grananja | 6,0 → 1,3 % | kraj petlje; kraća petlja = manje promašaja |
| Back-End Bound | instrukcija je spremna, ali čeka | 9,8 → 69,2 % | dijeli se na dvije podgrupe ispod |
| — Memory Bound | čeka podatke iz keša ili RAM-a | **6,1 → 41,9 %** | glavna brojka: prije ne čekamo memoriju, poslije čekamo |
| — Core Bound | čeka izvršne jedinice ili rezultat prethodne instrukcije | 3,7 → 27,2 % | zauzeti portovi za učitavanje i FMA |
| DRAM Bound | dio vremena kada se čeka baš RAM (ne keš) | **4,3 → 43,6 %** | podaci dolaze iz RAM-a, ne iz keša |
| Memory Bandwidth | dio DRAM čekanja zbog propusnosti (ne kašnjenja) | 90,9 % (snimak 07) | ograničava količina podataka u sekundi, a ne vrijeme jednog pristupa |

### VTune: vektorizacija i efikasnost

| Brojka | Značenje | scalar → avx2x4 |
|---|---|---|
| FP Scalar / FP Vector | udio skalarnih / vektorskih FP instrukcija | 36 / 0 % → 0,7 / 38 % |
| Vector Capacity Usage | koliko širine vektorskih jedinica se koristi | 6 → 48 % |
| CPI | taktova po instrukciji (IPC = 1 / CPI) | 0,28 → 0,85 |
| instrukcija po elementu | broj instrukcija za jedan od 128 brojeva vektora | 7,12 → 0,89 (8× manje) |
| taktova po elementu | broj taktova za jedan element | 1,98 → 0,75 (2,6× manje) |

Kako su izračunate brojke po elementu: Clockticks ili Instructions Retired iz VTune-a podijeljeno sa brojem obrađenih elemenata, 4 × 1000 × (130.769 + 256) × 128. Četiri prolaza su zagrijavanje + 3 ponavljanja; 130.769 skeniranih vektora i 256 centroida po upitu; 128 brojeva po vektoru.

### Zamke koje asistent može pitati

- **"Zašto je DRAM Bound (43,6 %) veći od Memory Bound (41,9 %), ako je dio njega?"** Različite jedinice. Memory Bound je postotak *pipeline slotova*, a DRAM Bound i njegove podkategorije postotak *taktova*. Na snimku 07 to piše desno od brojke ("of Pipeline Slots" i "of Clockticks"), pa se ne sabiraju direktno.
- **"CPI je porastao sa 0,28 na 0,85; zar to nije gore?"** CPI sam nije mjera brzine. Instrukcija ima 8× manje, a svaka u prosjeku traje 3× duže jer čeka memoriju. Ukupno je 2,6× manje taktova.
- **"Brojke na snimcima malo se razlikuju od tabele."** Snimci prikazuju Summary cijelog programa (sve niti i funkcije), a tabele su za samu funkciju udaljenosti. Npr. Retiring je 67,1 % na snimku 04, a 68,9 % u tabeli.
- **"Ako je zavisnost granica, zašto VTune ne kaže Core Bound?"** Dok sabiranje traje 2 takta, procesor izvrši ostalih 6 instrukcija petlje, pa VTune vidi koristan posao. Zavisnost se vidi u tačno 2 takta po elementu, a uklanjanjem lanca (`scalar4`) takti padaju za ~30 %.

## Ako bude demonstracija uživo

Uslov: na računaru postoje `data/sift` i sačuvani indeks. Prvo pokretanje pravi indeks i traje 1,5–2 minute, pa ga pokrenuti prije prezentacije.

Iz `implementacija/`, u PowerShellu ili cmd-u (vezano za P-jezgro):

```powershell
cmd /c start /b /wait /affinity 4 build\Release\prs_ann.exe ivf-bench data/sift 256 32 1000 5 scalar
```

```powershell
cmd /c start /b /wait /affinity 4 build\Release\prs_ann.exe ivf-bench data/sift 256 32 1000 5 avx2x4
```

U ispisu pokazati medijan, QPS i red "identicno skalarnoj: 1000 / 1000 upita". Skalarna verzija traje oko 1 minute (zagrijavanje + 5 ponavljanja po ~12 s). `avx2x4` traje oko 45 s, jer jednom izračuna i skalarnu referencu za poređenje. Ako nema vremena, staviti 2 ponavljanja umjesto 5.

VTune GUI: otvoriti rezultat iz `%LOCALAPPDATA%\prs-vtune\` (npr. `avx2x4-uarch-exploration`) i pokazati Summary.

## Ostala vjerovatna pitanja

| Pitanje | Kratak odgovor |
|---|---|
| Zašto ne `-ffast-math`? | Važi za cijeli fajl i mijenja pravila za float (redoslijed, NaN). `omp simd reduction` daje dozvolu samo za tu petlju. |
| Zašto AVX2 nije 8× brži? | Iz keša je blizu, 6×. Iz RAM-a ograničava propusnost memorije jednog jezgra (grafikon 11). |
| Je li 13,6 GB/s maksimum RAM-a? | Ne. Platforma ima ~45 GB/s, ali jedno jezgro može imati ograničen broj istovremenih zahtjeva. To je naša izmjerena granica uz Energy Saver (u Best performance 18,7 GB/s), ne potvrđena posebnim testom. |
| Da li VTune usporava program? | Malo: tokom snimanja 11,9–13,6 s naspram 11,7 s bez VTune-a. Zato su vremena u tabelama mjerena bez VTune-a. |
| Kako znate da su rezultati isti? | `ivf-bench` poredi svih 1000 upita sa skalarnom verzijom; testovi provjeravaju sve varijante za d od 1 do 130. SIFT vrijednosti su cijeli brojevi, pa su udaljenosti tačne u floatu. |
| Zašto je skalarna verzija bez vektorizacije? | Ona je referenca. Tako je i izmjereno koliko košta skalarni kod. |
| Da li je Energy Saver uticao? | Na apsolutna vremena jeste: u režimu Best performance sve je 1,3–1,4× brže (skalarna 9,06 s, `avx2x4` 3,59 s). Zaključci ostaju: isti poredak, ubrzanje 2,52×, granica memorije 18,7 GB/s i dalje ograničava (izvještaj §8, grafikon 12). |
| Šta dalje? | OpenMP (više jezgara), vektori kao `uint8` (4× manje bajtova), grupisanje upita po listi. |

Za detaljnije odgovore pogledati [izvještaj, §10](izvjestaj.md#10-moguća-pitanja).

## Prije prezentacije

- Laptop na punjaču, ostali programi zatvoreni (ako je demonstracija uživo).
- Otvoren izvještaj na GitHubu, ili lokalno u editoru sa Markdown pregledom.
- Ako je demonstracija uživo: indeks već napravljen, a komande isprobane jednom.
