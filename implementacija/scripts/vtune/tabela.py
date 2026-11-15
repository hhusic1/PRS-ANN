"""Tabele prije/poslije iz VTune CSV izvjestaja (docs/vtune/rezultati/).

Upotreba (iz korijena repoa):
    python implementacija/scripts/vtune/tabela.py [varijanta ...]

Bez argumenata koristi scalar, scalar4, avx2, avx2x4. Ispisuje Markdown tabele:
top-down metrike funkcije udaljenosti (uarch-exploration), memorijske metrike
(memory-access) i takte/instrukcije po elementu.

Broj obradjenih elemenata je isti za sve varijante (isti indeks i parametri):
(zagrijavanje + ponavljanja) x upiti x (skenirano po upitu + nlist) x d.
Podrazumijevano odgovara pokretanju iz collect.ps1: 4 x 1000 x (130769 + 256) x 128.
"""

import csv
import sys
from pathlib import Path

REZULTATI = Path(__file__).resolve().parents[3] / "docs" / "vtune" / "rezultati"
FUNKCIJA = {
    "scalar": "ann::l2_scalar",
    "scalar4": "ann::l2_scalar4",
    "auto": "ann::l2_auto",
    "omp": "ann::l2_omp",
    "avx2": "ann::l2_avx2_fma1",
    "avx2x4": "ann::l2_avx2_fma4",
}
PODRAZUMIJEVANE = ["scalar", "scalar4", "avx2", "avx2x4"]
ELEMENATA = 4 * 1000 * (130769 + 256) * 128


def citaj_funkcije(varijanta, analiza):
    putanja = REZULTATI / varijanta / f"{analiza}-functions.csv"
    with open(putanja, encoding="utf-8-sig", newline="") as f:
        return {red["Function"]: red for red in csv.DictReader(f)}


def dram_propusnost(varijanta):
    """(platforma max, izmjereni max, prosjek) u GB/s iz memory-access sazetka.

    Mjeri se cijeli sistem (uncore brojaci), ne samo nas program."""
    putanja = REZULTATI / varijanta / "memory-access-summary.csv"
    with open(putanja, encoding="utf-8-sig", newline="") as f:
        for red in csv.reader(f):
            if len(red) >= 5 and red[1] == "DRAM, GB/sec":
                return red[2], red[3], red[4]
    return "", "", ""


def kolona(red, *kraj):
    """Vrijednost P-core kolone cije ime zavrsava datim nizom (npr. 'Memory Bound(%)')."""
    for kljuc, vrijednost in red.items():
        if kljuc and "E-core" not in kljuc and kljuc.endswith(kraj):
            return vrijednost
    return ""


def broj(tekst, decimala=1):
    try:
        return f"{float(tekst):.{decimala}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "-"


def tabela(zaglavlje, redovi):
    print("| " + " | ".join(zaglavlje) + " |")
    print("|" + "---|" * len(zaglavlje))
    for red in redovi:
        print("| " + " | ".join(red) + " |")
    print()


def tabela_vremena(fajl, varijante):
    """Tabela iz 'CSV:' redova ivf-bench ispisa (samostalna mjerenja, bez VTune-a)."""
    putanja = REZULTATI / fajl
    if not putanja.exists():
        return
    redovi = {}
    for linija in putanja.read_text(encoding="utf-8").splitlines():
        if "CSV:" in linija:
            polja = linija.split("CSV:", 1)[1].strip().split(",")
            redovi[polja[0]] = polja
    if not redovi:
        return
    print(f"### Vrijeme pretrage ({fajl})\n")
    # sve varijante iz fajla, ustaljenim redom (i one bez VTune snimanja, npr. auto i omp)
    poredak = ["scalar", "scalar4", "auto", "omp", "avx2", "avx2x4"]
    sve = [v for v in poredak if v in redovi] + [v for v in redovi if v not in poredak]
    osnova = float(redovi["scalar"][6]) if "scalar" in redovi else None
    tabela(["Varijanta", "medijan (s)", "QPS", "ns po udaljenosti", "GB/s", "Recall@10", "ubrzanje"],
           [[v, broj(p[6], 3), broj(p[8]), broj(p[10]), broj(p[11]), broj(p[12], 4),
             (broj(osnova / float(p[6]), 2) + "×") if osnova else "-"]
            for v in sve for p in [redovi[v]]])


def main(varijante):
    tabela_vremena("vremena-sift1m.txt", varijante)
    tabela_vremena("vremena-sift10k-u-kesu.txt", varijante)

    print("### Udio CPU vremena (hotspots)\n")
    redovi = []
    for v in varijante:
        funkcije = citaj_funkcije(v, "hotspots")
        ukupno = sum(float(r["CPU Time"]) for r in funkcije.values())
        udio = lambda ime: broj(100 * float(funkcije[ime]["CPU Time"]) / ukupno) if ime in funkcije else "-"
        redovi.append([v, udio(FUNKCIJA[v]), udio("ann::IvfFlatSeq::search")])
    tabela(["Varijanta", "funkcija udaljenosti %", "IvfFlatSeq::search %"], redovi)

    uarch = {v: citaj_funkcije(v, "uarch-exploration")[FUNKCIJA[v]] for v in varijante}
    memorija = {v: citaj_funkcije(v, "memory-access").get(FUNKCIJA[v], {}) for v in varijante}
    dram = {v: dram_propusnost(v) for v in varijante}

    metrike = [
        ("CPI", ("CPI Rate",), 3),
        ("Retiring %", ("Retiring(%)",), 1),
        ("— FP Scalar %", ("FP Scalar(%)",), 1),
        ("— FP Vector %", ("FP Arithmetic:FP Vector(%)",), 1),
        ("Front-End Bound %", ("Front-End Bound(%)",), 1),
        ("Bad Speculation %", ("Bad Speculation(%)",), 1),
        ("Back-End Bound %", ("Back-End Bound(%)",), 1),
        ("— Memory Bound %", ("Back-End Bound:Memory Bound(%)",), 1),
        ("—— L1 Bound %", ("Memory Bound:L1 Bound(%)",), 1),
        ("—— L3 Bound %", ("Memory Bound:L3 Bound(%)",), 1),
        ("—— DRAM Bound %", ("Memory Bound:DRAM Bound(%)",), 1),
        ("— Core Bound %", ("Back-End Bound:Core Bound(%)",), 1),
        ("Vector Capacity Usage %", ("Vector Capacity Usage (FPU)(%)",), 1),
    ]
    print(f"### Top-down metrike funkcije udaljenosti (uarch-exploration, P-jezgro)\n")
    tabela(["Metrika"] + varijante,
           [[ime] + [broj(kolona(uarch[v], *kraj), dec) for v in varijante] for ime, kraj, dec in metrike])

    print("### Takti i instrukcije po elementu\n")
    redovi = []
    for ime, kljuc in (("taktova po elementu", "Clockticks"), ("instrukcija po elementu", "Instructions Retired")):
        redovi.append([ime] + [broj(float(uarch[v][kljuc]) / ELEMENATA, 2) for v in varijante])
    tabela(["Metrika"] + varijante, redovi)

    print("### Memorija (memory-access)\n")
    redovi = [
        ["Memory Bound funkcije %"] + [broj(kolona(memorija[v], "Memory Bound(%)")) for v in varijante],
        ["DRAM Bound funkcije %"] + [broj(kolona(memorija[v], "Memory Bound:DRAM Bound(%)")) for v in varijante],
        ["DRAM GB/s, prosjek (sistem)"] + [broj(dram[v][2]) for v in varijante],
        ["DRAM GB/s, izmjereni max (sistem)"] + [broj(dram[v][1]) for v in varijante],
        ["DRAM GB/s, max platforme"] + [broj(dram[v][0]) for v in varijante],
    ]
    tabela(["Metrika"] + varijante, redovi)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:] or PODRAZUMIJEVANE)
