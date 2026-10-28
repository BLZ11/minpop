# MinPop: Minimal Basis Set Population Analysis

**Chemically readable atomic charges, spin densities and σ/π/δ/φ bond populations from quantum chemistry calculations, for any element in the periodic table, with Hartree-Fock, DFT or GVB-PP wavefunctions. Built on PySCF. Reproduces Gaussian 16's MBS population analysis.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PySCF](https://img.shields.io/badge/PySCF-2.3+-green.svg)](https://pyscf.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Contents

1. [What MinPop does](#1-what-minpop-does) (for everyone)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Reading the output](#4-reading-the-output) (for everyone)
5. [Drivers and options](#5-drivers-and-options)
6. [Method](#6-method) (for specialists)
7. [Choosing the minimal basis](#7-choosing-the-minimal-basis)
8. [σ, π, δ, φ bond populations](#8-σ-π-δ-φ-bond-populations)
9. [GVB-PP wavefunctions](#9-gvb-pp-wavefunctions)
10. [Python API](#10-python-api)
11. [Validation](#11-validation)
12. [References](#12-references)

---

## 1. What MinPop does

A quantum chemistry calculation gives a wavefunction, not charges or bonds. Mulliken population analysis turns the wavefunction into atomic charges and bond populations, but it behaves badly in the large, flexible basis sets that accurate calculations need: a diffuse function sitting on one atom can describe electrons that physically belong to its neighbour, and Mulliken books them to the wrong atom.

MinPop fixes this by first projecting the occupied orbitals onto a *minimal* basis, one function per occupied atomic orbital, and running the Mulliken analysis there. With only one function per orbital, nothing can wander. The resulting numbers read like a chemist's description of the molecule: a charge per atom, a spin per atom, a population per orbital, and a shared population for every bond. This is the "MBS" analysis that Gaussian 16 prints with `Pop=(Full) IOp(6/27=122,6/12=3)`, and MinPop reproduces it to six decimals.

Version 2 extends it beyond what Gaussian offers:

| You want to | MinPop gives you |
|---|---|
| analyse transition metals, lanthanides, actinides, all-electron and relativistic | the ANO-RCC-MB minimal basis (H to Cm) and an X2C scalar-relativistic Hamiltonian |
| keep heavy-atom accuracy without paying for it on H, C, N, O | a mixed computational basis: ANO-RCC on the metals, 6-31G*/CBSB3/def2 on the light atoms |
| know whether a bond is σ, π or δ, and which part the unpaired electrons occupy | every bond population and spin-density bond population resolved into σ, π, δ, φ, printed as matrices |
| analyse a correlated (GVB perfect-pairing) wavefunction | the same report from GVB-PP natural orbitals, plus a per-pair bond table |
| know how much the minimal-basis picture misses | a "projection completeness" line with the fraction of density the minimal basis could not represent |
| compare against Gaussian | the STO-3G path, unchanged and validated |

---

## 2. Installation

```bash
git clone https://github.com/BLZ11/minpop.git
cd minpop
pip install .            # or: pip install -e .   (editable; needs pip >= 21.3)
```

Dependencies: NumPy, SciPy, PySCF ≥ 2.3 (installed automatically).

The scripts also run without installing: `python minpop/minpop_rohf.py ...`; each driver finds its sibling modules by itself.

Optional:

- **GVB-PP driver**: `git clone https://github.com/keithgroup/gvbpp.git && pip install ./gvbpp` (or add the clone to `PYTHONPATH`).
- **CBSB3 / CBSB7 bases**: copy `cbsb3_basis_pyscf.py` and `cbsb7_basis_pyscf.py` into `minpop/basis_sets/`.
- On a cluster with an ASCII locale, `export PYTHONIOENCODING=utf-8` so the σ π δ φ symbols print.

---

## 3. Quick start

```bash
# 1. Gaussian-MBS convention (STO-3G projection, 6-31+G calculation): triplet CH2 with UHF
minpop-uhf -xyz examples/xyz/ch2.xyz -mult 3

# 2. A transition-metal diatomic: Pt2 triplet, PBE, all-electron, relativistic,
#    ANO-RCC-MB projection, bonds resolved into σ/π/δ
minpop-uks -xyz examples/xyz/pt2.xyz -mult 3 -xc pbe \
           -basis ano-rcc-vdzp -minimal-basis ano-rcc-mb -x2c -symmetry

# 3. Same, with a cheap basis on the light atom
minpop-uks -xyz examples/xyz/pto.xyz -mult 3 -xc pbe \
           -basis ano-rcc-vdzp -basis-light "6-31G*" -minimal-basis ano-rcc-mb -x2c -symmetry-shells

# 4. GVB perfect pairing, six pairs for the Cr2 sextuple bond, then MinPop
minpop-gvbpp -xyz examples/xyz/cr2.xyz -mult 1 -npairs 6 \
             -basis ano-rcc-vdzp -minimal-basis ano-rcc-mb -x2c -symmetry-shells -soscf -gvb-trials 8
```

Geometries are plain `.xyz` files (Å). Multiplicity is `2S+1`.

---

## 4. Reading the output

A report has the following blocks, in this order. The first six match Gaussian's MBS output line for line.

```
Minimal-basis projection: ANO-RCC-MB (all-electron, generic (n,l,m) order)
 Projection completeness (alpha): 78.97 of 79 electrons captured, leak 0.029 (0.037%); least-captured MO #69 at 97.8%
============================================================
MinPop Analysis (UKS)
============================================================
     Alpha  MBS Density Matrix:            ┐ the projected density, per orbital pair
     Beta  MBS Density Matrix:             ┘ (specialists; most readers skip these)
     Full MBS Mulliken population analysis   population per orbital pair
     MBS Gross orbital populations           electrons in each atomic orbital: Total, Alpha, Beta, Spin
          MBS Condensed to atoms (all electrons)   diagonal: electrons on each atom; off-diagonal: shared in each bond
          MBS Atomic-Atomic Spin Densities        same for the spin density
 MBS Mulliken charges and spin densities         one line per atom: charge, spin
 Sum of MBS Mulliken charges
============================================================
MBS bond populations resolved by symmetry (σ, π, δ, φ)    [with -symmetry]
```

**For practitioners, the four numbers that matter most:**

- **Charge** (`MBS Mulliken charges`): partial charge on each atom. Pt in PtO: +0.36.
- **Spin** (same block): unpaired electrons on each atom. Pt in PtO: 1.02, O: 0.98 in the triplet.
- **Bond population** (`Condensed to atoms`, off-diagonal): electrons shared between two atoms. Larger means more covalent sharing. Pt–O: 0.25, Pt–H: 0.33. This measures *shared* density only; a polar bond with a large ionic contribution shows a small number here and a large charge separation above. It is not a bond order.
- **Orbital populations** (`Gross orbital populations`): the configuration of each atom as the molecule sees it. Pt in Pt₂: 5d 8.85, 6s 1.10, 6p 0.06, i.e. 5d⁹6s¹.

**The symmetry section** (`-symmetry`) splits each bond population and each spin-density bond population into σ, π, δ and φ parts and prints them as the same kind of matrices. Reading PtO, ³Σ⁻, PBE: Pt–O total 0.248 = σ 0.110 + π 0.138; spin density π −0.172. In words: the bond has more π than σ sharing, and the two unpaired electrons sit in the π antibonding combination (the negative sign). `-symmetry-shells` adds the same matrices per shell, so you can see that the π part is 5d–2p (0.094) and 6p–2p (0.049).

**Projection completeness** tells you how much of the occupied density had no room in the minimal basis before the electron count was restored. It is a few percent on lone pairs and SOMOs for STO-3G, and below 0.1 % for heavy atoms with ANO-RCC-MB. A large leak would mean the minimal-basis picture is missing something.

---

## 5. Drivers and options

| Command | Script | Wavefunction | Spin density from |
|---|---|---|---|
| `minpop-rohf` | `minpop_rohf.py` | RHF / ROHF | singly occupied orbitals |
| `minpop-uhf` | `minpop_uhf.py` | UHF with Löwdin spin annihilation | α − β after annihilation |
| `minpop-rks` | `minpop_rks.py` | RKS / ROKS, any libxc functional (`-xc`) | singly occupied orbitals |
| `minpop-uks` | `minpop_uks.py` | UKS | α − β |
| `minpop-gvbpp` | `minpop_gvbpp.py` | GVB perfect pairing (gvbpp) | open shells |
| `minpop-export` | `minpop_json_csv.py` | | JSON / CSV export of finished reports |

| Option | Meaning |
|---|---|
| `-xyz FILE -charge N -mult M` | geometry (Å), charge, multiplicity 2S+1 |
| `-basis NAME` | computational basis: any PySCF name, `cbsb3`, `cbsb7`, `ano-rcc-vdzp` / `-vtzp` / `-vqzp` / `-mb`, a `.py` basis file, or `module:DICT` |
| `-basis-light NAME`, `-light-max-z Z` | basis for atoms with Z ≤ 18 (Ar); `-basis` then covers only the heavier atoms. ECPs are detected per side |
| `-minimal-basis NAME` | projection target: `sto-3g` (default), `ano-rcc-mb`, `ano-rcc-mb-occ`, `sto-3g-ano`, `sto-3g-ano-occ`, `minao`, or `LIGHT+HEAVY` such as `sto-3g+ano-rcc-mb` or `sto-3g+minao` (LIGHT on Z ≤ `-light-max-z`, HEAVY on the rest); see §7 |
| `-x2c`, `-x2c-atomic` | scalar-relativistic X2C-1e Hamiltonian for the whole molecule; use with ANO-RCC and with any all-electron basis beyond Kr. `-x2c-atomic` (implies `-x2c`) builds the decoupling atom by atom (DLU), same populations, cost linear in the number of atoms |
| `-symmetry`, `-symmetry-shells` | σ/π/δ/φ matrices condensed to atoms, and additionally to shells (`-sigma-pi` is an accepted alias) |
| `-label-style {gaussian, physical}` | orbital labels as Gaussian prints them, a running shell counter (Si 3d = 4D, Pt 5d = 9D), or by principal quantum number (3D, 5D). Default: Gaussian's for `sto-3g` and `sto-3g-ano`, physical for `ano-rcc-mb`. Labels only |
| `-soscf`, `-guess`, `-stable`, `-guessmix`, `-max-cycle` | SCF convergence: Newton solver, initial guess, stability analysis, broken-symmetry mixing |
| `-json FILE`, `-csv-long FILE` | machine-readable export |
| `-npairs N`, `-gvb-trials N`, `-gvb-seed N`, `-gvb-window N`, `-no-keep X` | GVB-PP driver only; see §9 |

`-h` on any driver lists everything.

---

## 6. Method

Let C be the occupied MO coefficients in the computational basis, S_min the overlap matrix of the minimal basis, and S_cross the overlap between the two bases. The projected orbitals are

```
C' = S_min⁻¹ · S_cross · C · M^(−1/2),      M = Cᵀ · S_crossᵀ · S_min⁻¹ · S_cross · C
```

M^(−1/2) is the symmetric (Löwdin) orthonormalization, which restores unit norm to each projected orbital so the minimal-basis density P = Σ n_i c'_i c'_iᵀ holds exactly N electrons. The Mulliken analysis then proceeds in the minimal basis: gross populations are the diagonal of P S_min, bond populations the summed off-diagonal blocks. This is the MBS procedure of Montgomery et al. (refs 1, 2) as implemented in Gaussian.

The eigenvalues of M are the captured fractions of the occupied orbitals (1 = fully representable). N − tr M per spin is the density outside the minimal-basis span, reported as "Projection completeness". It falls on valence orbitals, never on cores, and does not grow when diffuse functions are added to the computational basis, which is the property the method was designed for.

Spin: ROHF/ROKS spin density comes from the singly occupied orbitals; UHF/UKS from P_α − P_β, with Löwdin annihilation of the first spin contaminant in the UHF driver; GVB-PP from the open shells only (§9).

Relativity: `-x2c` applies the spin-free exact-two-component one-electron Hamiltonian to the whole molecule. The projection is unchanged, since X2C orbitals live in the same non-relativistic AO basis. Relativity cannot be switched on for one element only: the Hamiltonian carries no element label, and the effect is large only near heavy nuclei anyway. `-x2c-atomic` builds the decoupling atom by atom (the DLU approximation of Peng and Reiher), which makes the X2C step linear in the number of atoms; on PtO it reproduces full X2C populations to 10⁻⁴. An ECP on the metal is the other way to confine relativity to the metal.

Orientation: geometries are put into Gaussian's standard orientation, and an azimuthal gauge fixes the orientation of degenerate π/δ components in linear molecules, so orbital-resolved matrices are reproducible and comparable with Gaussian's.

---

## 7. Choosing the minimal basis

The minimal basis defines the partitioning. Any choice is a convention; hold it fixed within a study.

| `-minimal-basis` | Elements | What it is |
|---|---|---|
| `sto-3g` (default) | H to Xe | Gaussian's MBS convention: STO-3G on H to Ne, STO-3G* on Na to Ar (adds a 3d shell, five spherical functions), STO-3G on K to Xe. Three Gaussians fit to Slater orbitals with Slater's-rules exponents, slightly contracted relative to free atoms. The validated path |
| `ano-rcc-mb` | H to Cm | One contracted atomic natural orbital per occupied atomic orbital, plus the valence p on metals (C: 2s1p, Fe: 4s3p1d, Pt: 6s5p3d1f). All-electron, contracted for a scalar-relativistic Hamiltonian, so it covers the whole periodic table and needs no ECP. The only all-electron minimal basis beyond Xe |
| `ano-rcc-mb-occ` | H to Cm | ANO-RCC-MB without the empty outermost p on s-, d- and f-block atoms beyond Ar (4p on Sc–Zn, 5p on Y–Cd, 6p on Hf–Hg and the lanthanides); p-block atoms and H–Ar unchanged. This is MINAO's valence shell structure wherever MINAO is defined, extended by the same rule to the elements it lacks. Use it when a metal charge should not absorb ligand density through a diffuse empty p: Rh(CO)₃ puts 0.75 e into Rh 5p with `ano-rcc-mb` and reads Rh +0.04 instead of +0.7 |
| `sto-3g-ano` | H to Cm | The `sto-3g` rule for main-group atoms up to Ar (STO-3G on H to Ne, STO-3G* on Na to Ar) and ANO-RCC-MB on every heavier atom, transition metals included. Same labels and numbers as `sto-3g` for molecules of H to Ar; the default choice for metal complexes with main-group ligands. Alias of `sto-3g+ano-rcc-mb` |
| `minao` | H to Kr all-electron; Y–Cd, In–Xe, Hf–Hg, Tl–Rn as valence-only sets | PySCF's IAO reference basis. Kept for completeness and for ECP calculations on 4d/5d metals, where it is the only valence-matched choice; not recommended as a general minimal basis (see below) |

How the two main choices differ, on the same HF/ANO-RCC-VDZP wavefunctions:

| Bond | q(A), STO-3G | q(A), ANO-RCC-MB |
|---|---|---|
| H₃C–CH₃ | −0.18 | −0.43 |
| H–OH | −0.66 | −0.94 |
| C≡O | +0.33 | +0.51 |
| Li–F | +0.66 | +0.89 |

Nonpolar bonds partition the same way (H–H bond population 0.397 vs 0.417). Polar bonds read more ionic with ANO-RCC-MB, because an ANO is a faithful free-atom orbital with the correct long-range tail (H 1s ⟨r⟩ 1.38 bohr, vs 1.21 for STO-3G), so it reaches the neighbour and claims density that the compact STO-3G function cannot. Neither basis contains diffuse functions; both stay at one function per occupied orbital. ANO-RCC-MB loses less density in the projection (0.8 % vs 2.1 % for H₂O; 0.03 % for Pt₂).

The general form is `-minimal-basis LIGHT+HEAVY`: `sto-3g+ano-rcc-mb` is what `sto-3g-ano` expands to, and `sto-3g+minao` pairs STO-3G on the light atoms with MINAO's valence-only functions on an ECP-bearing metal (def2-ECP on Rh leaves 4s4p4d5s, which is exactly MINAO's 2s1p1d; the shells are labelled 4S, 4P, 4D, 5S, with the ECP core shells counted in). The cutoff is `-light-max-z`, shared with `-basis-light`. `examples/run_rhco3.sh` runs Rh(CO)₃ through five such combinations.

Rule of thumb: `sto-3g` when results must be compared with existing STO-3G/Gaussian numbers; `sto-3g-ano` for metal complexes, where the ligands keep the STO-3G convention and the metal gets the all-electron ANO function set (identical to `sto-3g` on SiH₄ and HCl); `ano-rcc-mb` when every bond in a study, light or heavy, should be partitioned by one convention. For a homonuclear or heavy–heavy bond the two agree.

The minimal basis must match the core treatment of the calculation. With an ECP on the metal (def2 family), use MINAO there (`sto-3g+minao`): it is valence-only and fits the 28- or 60-electron core exactly. An all-electron minimal basis on an ECP atom (ANO-RCC-MB, or STO-3G as Gaussian does) carries core functions the calculation has no electrons for, and the nodeless pseudo-orbitals spill into them; Rh/def2-ECP projected onto ANO-RCC-MB puts about 1.3 e into 3s3p3d and the charges become meaningless. The driver prints a warning in that case. Conversely, ANO-RCC-MB belongs with an all-electron calculation (`-basis ano-rcc-vtzp -x2c`), which is where `-x2c` is also required.

**How the minimal bases differ on H–Ar.** They agree on H, He and B–Ne (1s; 2s 1p), and nowhere else:

| Elements | STO-3G / STO-3G* | MINAO | ANO-RCC-MB |
|---|---|---|---|
| Li, Be | 2s 1p | 2s | 2s 1p |
| Na, Mg | 3s 2p 1d | 3s 1p | 3s 2p |
| Al–Ar | 3s 2p 1d | 3s 2p | 3s 2p |

STO-3G* carries a 3d polarization shell on the second row (14 functions on Si against 9 or 6), which neither of the others has, so second-row populations are not interchangeable across the three. MINAO drops the empty valence p on Li, Be, Na, Mg as it does on the metals; `ano-rcc-mb-occ` leaves H–Ar as in ANO-RCC-MB.

**On MINAO.** It is available because PySCF ships it and because an ECP calculation on a 4d or 5d metal has no other valence-matched minimal basis. It is not a good general choice: its light-atom functions come from a different source (cc-pVTZ atomic orbitals) than its heavy-atom functions, it has no entries for K, Rb, Sr, Cs, Ba, the lanthanides or actinides, beyond Kr it is valence-only and cannot be used with an all-electron calculation, and it leaves out the 3d of STO-3G* on Na–Ar. For light atoms it partitions like ANO-RCC-MB (H₂O, CO, LiF within 0.01 of it). Prefer `sto-3g`, `sto-3g-ano(-occ)` or `ano-rcc-mb(-occ)` unless the calculation uses an ECP, and then use `sto-3g+minao` and read the completeness line.

**ECPs and ANO-RCC-MB.** Removing the ECP's core shells from ANO-RCC-MB (keeping 4s 5s 4p 4d on Rh with the 28-electron def2-ECP) is possible but projects poorly: the all-electron ANO valence functions have radial nodes inside the core where the pseudo-orbitals are nodeless, and on RhH 7.5 % of the valence density leaks, against 1.6 % with MINAO. MinPop therefore does not offer it. With an ECP use MINAO on that atom; with ANO-RCC-MB use an all-electron calculation.

The ANO-RCC family (`minpop/basis_sets/anorcc_basis_pyscf.py`) is sliced from PySCF's built-in ANO-RCC general contraction with per-element column counts; the result equals Basis Set Exchange's ano-rcc-mb/vdzp/vtzp/vqzp to 10⁻¹². No extra package is needed at run time.

---

## 8. σ, π, δ, φ bond populations

`-symmetry` appends a section after the Mulliken summary. For each pair of atoms, the AO blocks of P and S are rotated into the frame whose z′ axis lies along the bond (Wigner D matrices for real spherical harmonics, any angular momentum). In that frame the two-centre overlap is diagonal in the magnetic quantum number, so each term P′S′ belongs to exactly one class: σ (m = 0), π (|m| = 1), δ (|m| = 2), φ (|m| = 3). The classes sum to the off-diagonal "Condensed to atoms" element exactly; the same holds for the spin density. Same-atom terms have no bond axis and no class, so they appear only in the full matrices. No analytic overlap formulas, no fitted exponents, no renormalization; invariant under rotation of the molecule to 10⁻⁹; the diatomic selection rule is checked on every pair.

Printed, in the regular MinPop matrix formats: `MBS σ Condensed to atoms (all electrons)`, `MBS σ Atomic-Atomic Spin Densities`, and with `-symmetry-shells` the matrices condensed to shells (rows 1S, 2S, 2P, ... in aufbau order), for each class. A class with no contribution (φ in most molecules) prints as one line.

Three platinum diatomics, UKS/PBE, ANO-RCC-VDZP, X2C, ANO-RCC-MB, experimental bond lengths:

| | total | σ | π | δ | spin σ | spin π |
|---|---|---|---|---|---|---|
| Pt₂ ³Σg⁻ | 0.406 | 0.293 | 0.118 | −0.005 | +0.011 | −0.103 |
| PtO ³Σ⁻ | 0.248 | 0.110 | 0.138 | 0 | −0.012 | −0.172 |
| PtH ²Δ | 0.331 | 0.331 | 0 | 0 | +0.001 | 0 |

Pt₂ bonds through 6s σ with 5dπ support; its closed δ shell nets slightly antibonding; the two unpaired electrons are π*. PtO is the reverse, more π than σ, with the unpaired electrons again in π*. PtH is a pure σ bond whose unpaired electron is a Pt 5dδ orbital with no partner on hydrogen, so no spin appears in the bond at all.

Programmatic access: `bond_symmetry.bond_components(mol_min, dm, S)` for per-pair totals and shell-pair detail, `bond_symmetry.class_matrices(mol_min, dm, S)` for the matrices.

---

## 9. GVB-PP wavefunctions

GVB perfect pairing (gvbpp, ref 7) describes each bond or lone pair by two natural orbitals, a bonding one with occupation n_b and a correlating one with n_c (n_b + n_c = 2), singlet-coupled; cores keep 2, open shells 1. `minpop_gvbpp.py` runs RHF/ROHF, hands it to gvbpp, projects the natural orbitals weighted by occupation, and prints the regular report. The spin density comes from the open shells only; perfect pairing has none from the pairs, so for a closed-shell singlet such as Cr₂ the spin-density matrices are zero and the antiferromagnetic coupling shows up in n_c instead.

A minimal basis can hold a bond's correlating orbital (σ* of O–H lives in O 2p and H 1s) but not a lone pair's (that is a 3s/3d-like orbital), and in H₂O nine natural orbitals would not fit in seven functions. So natural orbitals are projected in order of decreasing occupation, and a correlating orbital is kept only if the part of it that is linearly independent of those already kept exceeds `-no-keep` (default 0.5). Otherwise its occupation folds back into its bonding partner: the electron count stays exact and that pair is treated as uncorrelated at the minimal-basis level. H₂O keeps the two O–H correlating orbitals (94 % independent) and merges the two lone-pair ones.

A pair table precedes the report:

```
 GVB-PP pairs:
  pair     n_bond     n_corr      S_ab  corr NO     pair on       gross(A)   gross(B)  MBS bond pop   sym
     1    1.89460    0.10540    0.6183  kept 100%   Cr1-Cr2         1.00000    1.00000      0.197337   σ
     2    1.75973    0.24027    0.4604  kept  99%   Cr1-Cr2         1.00000    1.00000     -0.100252   σ
     3    1.80767    0.19233    0.5081  kept 100%   Cr1-Cr2         1.00001    0.99999      0.087729   π
     5    1.39785    0.60215    0.2100  kept 100%   Cr1-Cr2         1.00000    1.00000      0.025302   δ
```

For each pair: the occupations, the GVB overlap S_ab between the two singly occupied pair orbitals, whether the correlating orbital was projected, the atoms the pair joins (or "lone pair"), and the pair's own contribution to the bond population with its symmetry. Cr₂ reads as one real 4sσ bond, two moderate π bonds, a 3dσ pair whose antibonding admixture outweighs its weak overlap, and two δ pairs that are a pair of singlet-coupled localized electrons.

Practicalities. The pair search is restricted to valence orbitals (`-gvb-window`, automatic: Ar core frozen for K to Kr, Kr core for Rb to Xe, Xe + 4f for Cs to Rn), so semicore 3s/3p orbitals are never correlated in place of a bond; a pair that still looks core-like is flagged. Perfect-pairing energy surfaces of metal dimers have several minima: use `-gvb-trials 8` with two `-gvb-seed` values and keep the lowest energy. For Cr₂, nineteen of twenty starts converge to a solution 36 kcal/mol above the one with paired δ electrons.

GVB-PP contains pair (static) correlation and no dynamic correlation. For Cr₂ at 1.679 Å, same basis and projection:

| Cr–Cr bond population | RHF | PBE | GVB-PP |
|---|---|---|---|
| σ | 0.390 | 0.389 | 0.106 |
| π | 0.317 | 0.269 | 0.195 |
| δ | 0.147 | 0.147 | 0.051 |

A single determinant fills every bonding orbital and empties every antibonding one, so its bond populations are the maximal ones. The RHF-to-PBE shift in π tracks the fraction of exact exchange (PBE0 0.281, PBE exchange only 0.267): it is static correlation mimicked by GGA exchange, not the correlation functional. δ is fixed by symmetry for any single determinant.

---

## 10. Python API

```python
from minpop import run_uks_from_xyz, bond_components

res = run_uks_from_xyz("examples/xyz/pto.xyz", multiplicity=3, xc="pbe",
                       basis="ano-rcc-vdzp", minimal_basis="ano-rcc-mb", x2c=True,
                       verbose=False)
res["mulliken_charges"], res["spin_populations"], res["condensed_to_atoms"]

# from an existing PySCF calculation
from pyscf import gto, scf
from minpop.minpop_uhf import minpop_uhf
mol = gto.M(atom="O 0 0 0; O 0 0 1.2", basis="6-31G*", spin=2)
res = minpop_uhf(scf.UHF(mol).run(), minimal_basis="sto-3g")
```

Return dictionaries hold `dm_alpha`, `dm_beta`, `dm_total`, `dm_spin`, `pop_total`, `pop_spin`, `gross_orbital_pop`, `condensed_to_atoms`, `spin_atomic`, `mulliken_charges`, `spin_populations`, `ao_labels`, `ao_labels_raw`, `ao_reorder` (Gaussian order), and for GVB-PP `gvb_pairs`, `gvb_e_tot`, `gvb_e_corr`, `no_occupations_kept`.

---

## 11. Validation

- **STO-3G path versus Gaussian 16.** Mulliken charges, spin populations, gross orbital populations and condensed matrices match to six decimals for CH (quartet), CH₂ (triplet, UHF and ROHF), CHCl₃, TiO₂ (UHF and ROHF) and SnH₄; the Gaussian and PySCF reports are in `validation/`. Version 2 leaves this path byte-identical to 1.x apart from the added completeness line. Use Gaussian's standard-orientation geometry when comparing orbital-resolved matrices.
- **ANO-RCC truncations.** Sliced coefficients equal Basis Set Exchange's ano-rcc-mb/vdzp/vtzp/vqzp to 10⁻¹² for H, C, Si, Fe, Pt, U (`python minpop/basis_sets/anorcc_basis_pyscf.py` repeats the check with basis-set-exchange installed).
- **σ/π/δ/φ.** Classes sum to the overlap population to 10⁻¹²; invariant under rotation of the molecule; selection rule asserted on every pair; the generic orbital ordering reproduces the legacy Gaussian order on H₂O, SiH₄, ScH, CuCl, SnH₄.
- **GVB-PP.** H₂ reproduces CAS(2,2); H₂O keeps the O–H correlating orbitals and merges the lone-pair ones; triplet CH₂; CrH, CrO, Cr₂ in `examples/outputs/`.
- `pytest -q` runs the fast subset in about ten seconds.

Worked outputs for Pt₂, PtO, PtH (PBE) and CrH, CrO, Cr₂ (GVB-PP, RHF, PBE) are in `examples/outputs/`, produced by `examples/run_pt_pbe.sh` and `examples/run_cr_gvbpp.sh`.

---

## 12. References

1. Montgomery Jr., J. A.; Frisch, M. J.; Ochterski, J. W.; Petersson, G. A. *J. Chem. Phys.* **1999**, *110*, 2822. [10.1063/1.477924](https://doi.org/10.1063/1.477924)
2. Montgomery Jr., J. A.; Frisch, M. J.; Ochterski, J. W.; Petersson, G. A. *J. Chem. Phys.* **2000**, *112*, 6532. [10.1063/1.481224](https://doi.org/10.1063/1.481224)
3. Löwdin, P.-O. *Phys. Rev.* **1955**, *97*, 1509. [10.1103/PhysRev.97.1509](https://doi.org/10.1103/PhysRev.97.1509)
4. Roos, B. O.; Lindh, R.; Malmqvist, P.-Å.; Veryazov, V.; Widmark, P.-O. *J. Phys. Chem. A* **2004**, *108*, 2851 (ANO-RCC, main group). [10.1021/jp031064+](https://doi.org/10.1021/jp031064+)
5. Roos, B. O.; Lindh, R.; Malmqvist, P.-Å.; Veryazov, V.; Widmark, P.-O. *J. Phys. Chem. A* **2005**, *109*, 6575 (ANO-RCC, transition metals). [10.1021/jp0581126](https://doi.org/10.1021/jp0581126)
6. Saue, T. *ChemPhysChem* **2011**, *12*, 3077 (X2C). [10.1002/cphc.201100682](https://doi.org/10.1002/cphc.201100682)
7. Keith group, *gvbpp*: GVB perfect pairing for PySCF. [github.com/keithgroup/gvbpp](https://github.com/keithgroup/gvbpp)

## License

MIT. See [LICENSE](LICENSE).

## Author

Barbaro Zulueta, [BLZ11](https://github.com/BLZ11)
