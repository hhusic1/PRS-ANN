<#
Snima Intel VTune analize za komandu ivf-bench i izvozi CSV izvjestaje.

Pokretati kao ADMINISTRATOR (na Windowsu VTune hardversko uzorkovanje to trazi),
iz bilo kog foldera:
  powershell -ExecutionPolicy Bypass -File <repo>\implementacija\scripts\vtune\collect.ps1 -Distances scalar

Program mora biti build sa simbolima (-g), npr. CMake RelWithDebInfo:
  cmake -S . -B build/vtune -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
  cmake --build build/vtune

Rezultati:
  implementacija/build/vtune-results/<udaljenost>-<analiza>   sirovi rezultat (otvara se u VTune GUI-ju)
  docs/vtune/rezultati/<udaljenost>/<analiza>-summary.csv      sazetak analize
  docs/vtune/rezultati/<udaljenost>/<analiza>-functions.csv    metrike po funkcijama
  docs/vtune/rezultati/<udaljenost>/<analiza>-log.txt          ispis VTune-a i programa

Sirovi rezultati idu u kratku putanju (-ResultRoot, podrazumijevano %LOCALAPPDATA%\prs-vtune):
VTune-ov hardverski kolektor ne podrzava putanje duze od 260 znakova, a unutar foldera
rezultata pravi podfoldere i duga imena fajlova.

Putanja do programa ne smije imati razmake (start /affinity bi prvi navodnik shvatio kao naslov prozora).
-HotspotsMode sw: hotspots sa softverskim uzorkovanjem, radi i bez administratora (za probu skripte).
#>
param(
    [string[]]$Distances = @("scalar"),
    [string]$Exe = "build/vtune/prs_ann.exe",
    [string]$Dataset = "data/sift",
    [int]$Nlist = 256,
    [int]$Nprobe = 32,
    [int]$Queries = 1000,
    [int]$Repeats = 3,
    [string[]]$Analyses = @("hotspots", "uarch-exploration", "memory-access"),
    [int]$Cpu = 2,
    [string]$VTune = "C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe",
    [string]$ResultRoot = (Join-Path $env:LOCALAPPDATA "prs-vtune"),
    [ValidateSet("hw", "sw")][string]$HotspotsMode = "hw",
    [switch]$DryRun
)

# Ne "Stop": VTune pise napredak na stderr, a PowerShell 5.1 bi to pretvorio u gresku.
# Uspjeh svake komande se provjerava preko $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$impl = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$repo = (Resolve-Path (Join-Path $impl "..")).Path
Set-Location $impl

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
# Hardversko uzorkovanje trazi administratora; softverski hotspots ne trazi.
$needsHw = ($Analyses | Where-Object { $_ -ne "hotspots" }).Count -gt 0 -or $HotspotsMode -eq "hw"
if ($needsHw -and -not $admin -and -not $DryRun) {
    throw "Skriptu pokrenuti kao administrator (VTune hardversko uzorkovanje to trazi)."
}
New-Item -ItemType Directory -Force $ResultRoot | Out-Null
if (-not (Test-Path $Exe)) { throw "Ne postoji $Exe (napraviti build sa simbolima, -g)." }
if (-not (Test-Path $VTune)) { throw "Ne postoji $VTune." }

# Hibridni procesor (P i E jezgra): proces se veze za jedno logicko jezgro, da
# se mjerenje ne seli izmedju P i E jezgara. Na Intel 12. generaciji su P-jezgra
# prva logicka jezgra (0-7 kod i5-12500H).
$mask = "{0:X}" -f ([uint64]1 -shl $Cpu)
$exeFull = (Resolve-Path $Exe).Path

foreach ($dist in $Distances) {
    $outDir = Join-Path $repo "docs\vtune\rezultati\$dist"
    New-Item -ItemType Directory -Force $outDir | Out-Null
    foreach ($analysis in $Analyses) {
        $result = Join-Path $ResultRoot "$dist-$analysis"
        $knobs = @()
        if ($analysis -eq "hotspots") { $knobs = @("-knob", "sampling-mode=$HotspotsMode") }
        # Zadnji argument 0: bez poredjenja sa skalarnom referencom, jer bi
        # racunanje reference uslo u profil (provjera se radi bez VTune-a).
        $target = @("cmd.exe", "/c", "start", "/b", "/wait", "/affinity", $mask,
                    $exeFull, "ivf-bench", $Dataset, $Nlist, $Nprobe, $Queries, $Repeats, $dist, "0")
        # -data-limit 0: bez granice od 1000 MB (inace VTune prekine dugo snimanje).
        $collect = @("-collect", $analysis) + $knobs + @("-data-limit", "0", "-result-dir", $result, "--") + $target
        # Izvjestaji se prvo pisu pored rezultata (kratka putanja), pa kopiraju u docs.
        $summary = @("-report", "summary", "-result-dir", $result, "-format", "csv", "-csv-delimiter", "comma",
                     "-report-output", (Join-Path $ResultRoot "$dist-$analysis-summary.csv"))
        $functions = @("-report", "hotspots", "-result-dir", $result, "-group-by", "function", "-format", "csv",
                       "-csv-delimiter", "comma", "-report-output", (Join-Path $ResultRoot "$dist-$analysis-functions.csv"))

        Write-Host "=== $dist / $analysis (logicko jezgro $Cpu, maska 0x$mask) ==="
        if ($DryRun) {
            Write-Host "vtune $($collect -join ' ')"
            Write-Host "vtune $($summary -join ' ')"
            Write-Host "vtune $($functions -join ' ')"
            continue
        }
        if (Test-Path $result) { Remove-Item -Recurse -Force $result }
        # Tee-Object u PowerShellu 5.1 pise UTF-16, pa se log upisuje posebno, u UTF-8.
        & $VTune @collect 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable lines
        $code = $LASTEXITCODE
        $lines | Out-File -Encoding utf8 (Join-Path $outDir "$analysis-log.txt")
        if ($code -ne 0) { throw "VTune snimanje nije uspjelo ($dist / $analysis), kod $code" }
        & $VTune @summary 2>&1 | Out-Null
        & $VTune @functions 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "VTune izvjestaj nije uspio ($dist / $analysis), kod $LASTEXITCODE" }
        Copy-Item (Join-Path $ResultRoot "$dist-$analysis-summary.csv") (Join-Path $outDir "$analysis-summary.csv") -Force
        Copy-Item (Join-Path $ResultRoot "$dist-$analysis-functions.csv") (Join-Path $outDir "$analysis-functions.csv") -Force
    }
}
Write-Host "Gotovo. CSV izvjestaji su u $repo\docs\vtune\rezultati\"
Write-Host "Sirovi rezultati (za VTune GUI) su u $ResultRoot"
