"""Grafikoni (SVG) za izvjestaj iz istih podataka kao tabela.py.

Upotreba (iz korijena repoa):
    python implementacija/scripts/vtune/grafikoni.py

Pravi docs/vtune/slike/09-ubrzanje.svg, 10-top-down.svg, 11-propusnost.svg i
12-rezim-napajanja.svg (Energy Saver naspram Best performance).
Bez dodatnih paketa: SVG se pise direktno, a GitHub ga prikazuje u Markdownu.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabela import FUNKCIJA, REZULTATI, citaj_funkcije, kolona  # noqa: E402

SLIKE = REZULTATI.parent / "slike"
PORADAK = ["scalar", "scalar4", "auto", "omp", "avx2", "avx2x4"]
OPIS = {
    "scalar": "scalar (početno)",
    "scalar4": "scalar4 (4 akumulatora)",
    "auto": "auto (kompajler)",
    "omp": "omp simd",
    "avx2": "avx2 (1 akumulator)",
    "avx2x4": "avx2x4 (4 akumulatora)",
}

# Validirana kategorijska paleta (redoslijed fiksan) i tokeni za tekst.
BOJE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
POVRSINA, TEKST, TEKST2, MREZA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FONT = "Segoe UI, Helvetica, Arial, sans-serif"


def zarez(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",")


def vremena(fajl):
    """{varijanta: (medijan s, GB/s)} iz 'CSV:' redova ivf-bench ispisa."""
    redovi = {}
    for linija in (REZULTATI / fajl).read_text(encoding="utf-8").splitlines():
        if "CSV:" in linija:
            p = linija.split("CSV:", 1)[1].strip().split(",")
            redovi[p[0]] = (float(p[6]), float(p[11]))
    return redovi


def svg(sirina, visina, tijelo):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{sirina}" height="{visina}" '
            f'viewBox="0 0 {sirina} {visina}" font-family="{FONT}">\n'
            f'<rect width="{sirina}" height="{visina}" rx="8" fill="{POVRSINA}"/>\n'
            + "\n".join(tijelo) + "\n</svg>\n")


def tekst(x, y, s, vel=13, boja=TEKST, sidro="start", tezina="normal", oreol=False):
    # oreol: obrub boje povrsine, da linija iza teksta ne presijeca slova
    o = f' stroke="{POVRSINA}" stroke-width="4" paint-order="stroke"' if oreol else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{vel}" fill="{boja}" '
            f'text-anchor="{sidro}" font-weight="{tezina}"{o}>{s}</text>')


def naslov(t, podnaslov):
    return [tekst(24, 34, t, 17, tezina="600"), tekst(24, 56, podnaslov, 12.5, TEKST2)]


def legenda(x, y, stavke):
    out = []
    for boja, ime in stavke:
        out.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{boja}"/>')
        out.append(tekst(x + 18, y, ime, 12.5, TEKST2))
        x += 26 + 7.2 * len(ime)
    return out


def traka_h(x0, x1, y, h, boja, r=4):
    """Vodoravna traka sa zaobljenim krajem podatka (pocetak ostaje na osi)."""
    if r == 0 or x1 - x0 <= r:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 0):.1f}" height="{h}" fill="{boja}"/>'
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{boja}"/>')


def traka_v(x, w, yb, yt, boja, r=4):
    """Uspravna traka sa zaobljenim vrhom (dno ostaje na osi)."""
    if yb - yt <= r:
        return f'<rect x="{x:.1f}" y="{yt:.1f}" width="{w}" height="{max(yb - yt, 0):.1f}" fill="{boja}"/>'
    return (f'<path d="M{x:.1f},{yb:.1f} V{yt + r:.1f} A{r},{r} 0 0 1 {x + r:.1f},{yt:.1f} '
            f'H{x + w - r:.1f} A{r},{r} 0 0 1 {x + w:.1f},{yt + r:.1f} V{yb:.1f} Z" fill="{boja}"/>')


def grafikon_ubrzanje(ram, kes):
    S, lijevo, desno, vrh = 780, 190, 60, 104
    h, razmak, grupa = 14, 2, 46
    xmax = 7.0
    sx = lambda v: lijevo + (S - lijevo - desno) * v / xmax
    visina = vrh + grupa * len(PORADAK) + 40
    t = naslov("Ubrzanje pretrage u odnosu na skalarnu verziju (veće je bolje)",
               "1000 upita, nprobe 32, jedno P-jezgro, režim Energy Saver; sve varijante daju identične rezultate")
    t += legenda(24, 84, [(BOJE[0], "SIFT1M, podaci iz RAM-a"), (BOJE[1], "SIFT10K, podaci u kešu")])
    dno = vrh + grupa * len(PORADAK) - 8
    for v in range(0, int(xmax) + 1):
        x = sx(v)
        t.append(f'<line x1="{x:.1f}" y1="{vrh - 6}" x2="{x:.1f}" y2="{dno}" stroke="{MREZA}" stroke-width="1"/>')
        t.append(tekst(x, dno + 18, f"{v}×", 12, TEKST2, "middle"))
    for i, var in enumerate(PORADAK):
        y = vrh + i * grupa
        t.append(tekst(lijevo - 12, y + h + 5, OPIS[var], 13, TEKST, "end"))
        for j, (podaci, boja) in enumerate(((ram, BOJE[0]), (kes, BOJE[1]))):
            u = podaci["scalar"][0] / podaci[var][0]
            yy = y + j * (h + razmak)
            t.append(traka_h(sx(0), sx(u), yy, h, boja))
            t.append(tekst(sx(u) + 6, yy + h - 2.5, zarez(u, 2) + "×", 12, TEKST))
    return svg(S, visina, t)


def grafikon_top_down(uarch):
    S, lijevo, desno, vrh = 780, 150, 30, 120
    h, grupa = 40, 70
    sx = lambda p: lijevo + (S - lijevo - desno) * p / 100
    dijelovi = [
        ("Retiring (koristan posao)", "Retiring(%)"),
        ("Front-End Bound", "Front-End Bound(%)"),
        ("Bad Speculation", "Bad Speculation(%)"),
        ("Memory Bound (čeka memoriju)", "Back-End Bound:Memory Bound(%)"),
        ("Core Bound", "Back-End Bound:Core Bound(%)"),
    ]
    t = naslov("Gdje odlaze taktovi procesora: top-down podjela (VTune)",
               "funkcija udaljenosti, P-jezgro, ponovljeno snimanje na neopterećenom računaru, režim Energy Saver")
    t += legenda(24, 84, [(BOJE[k], ime) for k, (ime, _) in enumerate(dijelovi[:3])])
    t += legenda(24, 104, [(BOJE[k + 3], ime) for k, (ime, _) in enumerate(dijelovi[3:])])
    redovi = [("scalar", "scalar (prije)"), ("avx2x4", "avx2x4 (poslije)")]
    for i, (var, ime) in enumerate(redovi):
        y = vrh + 14 + i * grupa
        t.append(tekst(lijevo - 12, y + h / 2 + 5, ime, 13.5, TEKST, "end", "600"))
        vrijednosti = [float(kolona(uarch[var], kraj)) for _, kraj in dijelovi]
        ukupno = sum(vrijednosti)
        x = sx(0)
        for k, v in enumerate(vrijednosti):
            sirina = (S - lijevo - desno) * v / ukupno
            zadnji = k == len(vrijednosti) - 1
            x1 = x + sirina - (0 if zadnji else 2)
            t.append(traka_h(x, x1, y, h, BOJE[k], 4 if zadnji else 0))
            if sirina > 46:
                t.append(tekst(x + sirina / 2, y + h / 2 + 4.5, zarez(v) + " %", 12.5, TEKST, "middle"))
            elif k == 3:
                # uzak Memory Bound segment (pocetno stanje): natpis ispod trake, to je kljucna brojka
                cx = x + sirina / 2
                t.append(f'<line x1="{cx:.1f}" y1="{y + h + 2}" x2="{cx:.1f}" y2="{y + h + 9}" stroke="{TEKST}" stroke-width="1"/>')
                t.append(tekst(cx + 4, y + h + 21, f"Memory Bound {zarez(v)} %", 12.5, TEKST, "end"))
            x += sirina
    visina = vrh + 14 + grupa * len(redovi) + 10
    return svg(S, visina, t)


def grafikon_propusnost(ram, kes):
    S, lijevo, desno, vrh = 780, 60, 24, 104
    visina_osi, w, razmak = 240, 26, 2
    ymax = 40.0
    dno = vrh + visina_osi
    sy = lambda v: dno - visina_osi * v / ymax
    sirina_grupe = (S - lijevo - desno) / len(PORADAK)
    t = naslov("Propusnost skeniranja vektora (GB/s, veće je bolje)",
               "Energy Saver: iz RAM-a brze varijante zapnu na ~12–13,6 GB/s, iz keša ista funkcija čita do 35 GB/s")
    t += legenda(24, 84, [(BOJE[0], "SIFT1M, podaci iz RAM-a"), (BOJE[1], "SIFT10K, podaci u kešu")])
    for v in range(0, int(ymax) + 1, 10):
        y = sy(v)
        t.append(f'<line x1="{lijevo}" y1="{y:.1f}" x2="{S - desno}" y2="{y:.1f}" stroke="{MREZA}" stroke-width="1"/>')
        t.append(tekst(lijevo - 8, y + 4, str(v), 12, TEKST2, "end"))
    # referentna linija ide prva (iza traka i natpisa)
    granica = ram["avx2x4"][1]
    yg = sy(granica)
    t.append(f'<line x1="{lijevo}" y1="{yg:.1f}" x2="{S - desno}" y2="{yg:.1f}" stroke="{TEKST}" '
             f'stroke-width="1.5" stroke-dasharray="5 4"/>')
    t.append(tekst(lijevo + 6, yg - 24, f"najviše iz RAM-a na jednom jezgru (Energy Saver): {zarez(granica)} GB/s",
                   12, TEKST, tezina="600", oreol=True))
    for i, var in enumerate(PORADAK):
        cx = lijevo + sirina_grupe * (i + 0.5)
        for j, (podaci, boja) in enumerate(((ram, BOJE[0]), (kes, BOJE[1]))):
            x = cx - w - razmak / 2 + j * (w + razmak)
            gb = podaci[var][1]
            t.append(traka_v(x, w, dno, sy(gb), boja))
            t.append(tekst(x + w / 2, sy(gb) - 5, zarez(gb), 11.5, TEKST, "middle", oreol=True))
        t.append(tekst(cx, dno + 18, var, 12.5, TEKST, "middle"))
    return svg(S, dno + 34, t)


def grafikon_rezimi(es, bp):
    """Energy Saver naspram Best performance (SIFT1M): vrijeme i ubrzanje, dva panela."""
    S, lijevo, vrh = 780, 168, 128
    h, razmak, grupa = 12, 2, 38
    paneli = [  # (pocetak x, sirina, max vrijednost, podjela mreze, vrijednost, natpis)
        (lijevo, 236, 12.0, [0, 4, 8, 12], lambda p, v: p[v][0], lambda x: zarez(x, 2) + " s", "s"),
        (476, 236, 3.0, [0, 1, 2, 3], lambda p, v: p["scalar"][0] / p[v][0], lambda x: zarez(x, 2) + "×", "×"),
    ]
    t = naslov("Režim napajanja: Energy Saver naspram Best performance (SIFT1M)",
               "u Best performance sve je 1,3–1,4× brže, a ubrzanje i poredak varijanti ostaju isti")
    t += legenda(24, 84, [(BOJE[2], "Energy Saver (sva ostala mjerenja)"), (BOJE[3], "Best performance (kontrola)")])
    dno = vrh + grupa * len(PORADAK) - 8
    t.append(tekst(paneli[0][0], vrh - 16, "Vrijeme 1000 upita (manje je bolje)", 12.5, TEKST, tezina="600"))
    t.append(tekst(paneli[1][0], vrh - 16, "Ubrzanje u odnosu na scalar (veće je bolje)", 12.5, TEKST, tezina="600"))
    for x0, sir, vmax, mreza, _, _, jed in paneli:
        for m in mreza:
            x = x0 + sir * m / vmax
            t.append(f'<line x1="{x:.1f}" y1="{vrh - 6}" x2="{x:.1f}" y2="{dno}" stroke="{MREZA}" stroke-width="1"/>')
            t.append(tekst(x, dno + 18, f"{m}{' ' + jed if jed == 's' else jed}", 12, TEKST2, "middle"))
    for i, var in enumerate(PORADAK):
        y = vrh + i * grupa
        t.append(tekst(lijevo - 12, y + h + 5, OPIS[var], 13, TEKST, "end"))
        for x0, sir, vmax, _, vrijednost, natpis, _ in paneli:
            for j, (podaci, boja) in enumerate(((es, BOJE[2]), (bp, BOJE[3]))):
                vr = vrijednost(podaci, var)
                yy = y + j * (h + razmak)
                x1 = x0 + sir * vr / vmax
                t.append(traka_h(x0, x1, yy, h, boja))
                t.append(tekst(x1 + 5, yy + h - 2, natpis(vr), 11.5, TEKST))
    return svg(S, dno + 34, t)


def main():
    ram, kes = vremena("vremena-sift1m.txt"), vremena("vremena-sift10k-u-kesu.txt")
    uarch = {v: citaj_funkcije(v, "uarch-exploration")[FUNKCIJA[v]] for v in ("scalar", "avx2x4")}
    fajlovi = {
        "09-ubrzanje.svg": grafikon_ubrzanje(ram, kes),
        "10-top-down.svg": grafikon_top_down(uarch),
        "11-propusnost.svg": grafikon_propusnost(ram, kes),
        "12-rezim-napajanja.svg": grafikon_rezimi(ram, vremena("vremena-sift1m-best-performance.txt")),
    }
    for ime, sadrzaj in fajlovi.items():
        (SLIKE / ime).write_text(sadrzaj, encoding="utf-8")
        print("napisano:", SLIKE / ime)


if __name__ == "__main__":
    main()
