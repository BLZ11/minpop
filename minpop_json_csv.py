#!/usr/bin/env python3
"""
minpop_json_csv.py -- store MinPop outputs (PySCF or Gaussian) as customized JSON.

Why JSON and not cclib: cclib's data model covers the standard quantities every
code prints, but MinPop's minimal-basis-set outputs (MBS density matrices, gross
orbital populations, condensed-to-atoms, atomic spin densities, spin
contamination before/after annihilation) have no cclib attributes, so a cclib
bridge would mean writing a new parser library. This module parses the MinPop
report itself and writes one self-describing JSON record per calculation.

Standalone by design: it needs only numpy and the standard library. Import it
from anywhere, run it on a directory of finished jobs, or let minpop_rohf.py /
minpop_uhf.py call it through their -json flag. It reads both the PySCF MinPop
printout and a Gaussian .out, since both use the IOp(6/27=122) layout.

Where a field does have a cclib equivalent the mapping is:

    system.charge            -> cclib charge
    system.multiplicity      -> cclib mult
    system.natoms            -> cclib natom
    geometry.atomic_numbers  -> cclib atomnos
    geometry.input_orientation    -> cclib atomcoords[0]   (angstrom, as cclib)
    geometry.standard_orientation -> cclib atomcoords[-1]  (angstrom, as cclib)
    scf.energy_hartree       -> cclib scfenergies[-1]  (NOTE: cclib uses eV)

so a later cclib bridge is a rename, not a reparse.

Layout
------
Every quantity carries its own shape and units, and matrices are stored whole
(row-major nested lists: lower triangle, diagonal, and upper triangle) so a
record round-trips to the exact float values Gaussian printed.

    {
      "schema": "minpop-record", "schema_version": "1.0",
      "source":   {file, sha256, code, parsed_utc},
      "system":   {id, reaction, point, charge, multiplicity, natoms, nelectrons},
      "method":   {name, basis, guess_mix, stable_opt},
      "geometry": {units, atomic_numbers, atomic_symbols,
                   standard_orientation, input_orientation,
                   input_orientation_source, atom_order_consistent},
      "scf":      {energy_hartree, s2_before_annihilation, s2_after_annihilation,
                   n_alpha, n_beta},
      "mbs":      {basis_function_labels, gross, density_alpha, density_beta,
                   population_full, condensed_to_atoms, atomic_spin_densities,
                   mulliken, mulliken_sum}
    }

Usage
-----
    # one file -> one .json
    python minpop_json_csv.py --out rxn0081_point0_uhf_minpop.out

    # a whole tree -> one .json next to each .out
    python minpop_json_csv.py --runs pyscf_minpop --outdir json/

    # a whole tree -> a single JSON Lines dataset (best for ML consumers)
    python minpop_json_csv.py --runs pyscf_minpop --jsonl minpop_dataset.jsonl.gz

    # read back
    >>> from minpop_json import load_record
    >>> rec = load_record("rxn0081_point0_uhf_minpop.json")
    >>> rec["mbs"]["density_alpha"]["data"].shape      # numpy array (37, 37)
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "minpop-record"
SCHEMA_VERSION = "1.1"

_Z2SYM = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 35: "Br", 53: "I",
}


# --------------------------------------------------------------------------- #
# MinPop report parsing
#
# Self-contained on purpose: this module is the shared library for the MinPop
# tools (minpop_rohf.py / minpop_uhf.py -json call it, and any downstream script
# can import it), so it must not drag in a validation utility to read a report.
# It depends on nothing beyond numpy and the standard library.
#
# The readers below work on the Gaussian IOp(6/27=122) layout, which both the
# Gaussian .out and the PySCF MinPop printout use, so one code path serves both.
# Every quantity is taken from the LAST block, i.e. the final Stable=Opt pass.
# --------------------------------------------------------------------------- #


_INT = re.compile(r"^-?\d+$")

_FLT = re.compile(r"^[+-]?\d*\.\d+([eEdD][+-]?\d+)?$")

def _flt(t):
    """Parse one Gaussian/Fortran float token (handles D-exponents)."""
    return float(t.replace("D", "E").replace("d", "e"))

def _symmetrize(M):
    if M is None or M.shape[0] != M.shape[1]:
        return M
    L = np.tril(M)
    # if only the lower triangle was populated, mirror it
    if np.count_nonzero(np.triu(M, 1)) == 0:
        return L + L.T - np.diag(np.diag(L))
    return M


def _is_col_header(tokens):
    return bool(tokens) and all(_INT.match(t) for t in tokens)

def _floats(tokens):
    out = []
    for t in tokens:
        if _FLT.match(t):
            out.append(_flt(t))
    return out

def _read_blocked_matrix(lines, start):
    """Read a Gaussian column-blocked matrix beginning at line `start`.

    Handles both full and lower-triangular layouts (row's float count may be
    shorter than the block width). Returns (matrix, next_index). Stops at the
    first blank line or non-matrix line.
    """
    entries = {}
    max_r = max_c = -1
    i = start
    cur_cols = None
    while i < len(lines):
        raw = lines[i]
        toks = raw.split()
        if not toks:                      # blank line ends the matrix
            break
        if _is_col_header(toks):
            cur_cols = [int(t) - 1 for t in toks]
            i += 1
            continue
        if cur_cols is not None and _INT.match(toks[0]):
            r = int(toks[0]) - 1
            vals = _floats(toks[1:])      # skip leading row index + any labels
            if not vals:
                break
            for k, v in enumerate(vals):
                c = cur_cols[k]
                entries[(r, c)] = v
                max_r, max_c = max(max_r, r), max(max_c, c)
            i += 1
            continue
        break                             # not header, not data -> done
    if max_r < 0:
        return None, i
    M = np.zeros((max_r + 1, max_c + 1))
    for (r, c), v in entries.items():
        M[r, c] = v
    return M, i

def _find(lines, pattern, last=True):
    """Index of a matching line. Defaults to the LAST match, so with Stable=Opt
    (which prints multiple MBS blocks / SCF passes) we read the final one."""
    rx = re.compile(pattern)
    hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
    if not hits:
        return None
    return hits[-1] if last else hits[0]

def detect_method(text):
    if re.search(r"E\(ROHF\)", text) or re.search(r"\bROHF\b", text):
        return "rohf"
    if re.search(r"E\(UHF\)", text) or re.search(r"\bUHF\b", text):
        return "uhf"
    return None

def parse_gross(text):
    """MBS Gross orbital populations -> (orb_labels, array[N, k])."""
    lines = text.splitlines()
    i = _find(lines, r"MBS Gross orbital populations")
    if i is None:
        return None, None
    # next line is the Total/Alpha/Beta/Spin header
    j = i + 1
    if j < len(lines) and re.search(r"Total", lines[j]):
        j += 1
    rows, labels = [], []
    while j < len(lines):
        toks = lines[j].split()
        if not toks or not _INT.match(toks[0]):
            break
        vals = _floats(toks[1:])
        if not vals:
            break
        # orbital label = last non-float token (e.g. '1S', '2PX')
        lbl = None
        for t in reversed(toks[1:]):
            if not _FLT.match(t):
                lbl = t
                break
        labels.append(lbl)
        rows.append(vals)
        j += 1
    if not rows:
        return None, None
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows])
    return labels, arr

def parse_atomic_matrix(text, header):
    lines = text.splitlines()
    i = _find(lines, header)
    if i is None:
        return None
    M, _ = _read_blocked_matrix(lines, i + 1)
    return _symmetrize(M)

def parse_density_matrix(text, header):
    lines = text.splitlines()
    i = _find(lines, header)
    if i is None:
        return None
    M, _ = _read_blocked_matrix(lines, i + 1)
    return _symmetrize(M)

def parse_mulliken_charges_spins(text):
    """MBS Mulliken charges and spin densities -> array[N, 1 or 2]."""
    lines = text.splitlines()
    i = _find(lines, r"MBS Mulliken charges and spin densities:")
    if i is None:
        return None
    j = i + 1
    # skip the '1  2' column header line if present
    if j < len(lines) and _is_col_header(lines[j].split()):
        j += 1
    rows = []
    while j < len(lines):
        if "Sum of MBS" in lines[j]:
            break
        toks = lines[j].split()
        if not toks or not _INT.match(toks[0]):
            break
        vals = _floats(toks[1:])
        if not vals:
            break
        rows.append(vals)
        j += 1
    if not rows:
        return None
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows])

def parse_mbs_charge_sum(text):
    """The 'Sum of MBS Mulliken charges = <charge_sum> <spin_sum>' line -> [2].
    Uses the LAST such line (Stable=Opt prints several)."""
    ms = re.findall(r"Sum of MBS Mulliken charges\s*=\s*"
                    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)", text)
    if ms:
        c, s = ms[-1]
        return np.array([float(c), float(s)])
    ms1 = re.findall(r"Sum of MBS Mulliken charges\s*=\s*(-?\d+\.\d+)", text)
    return np.array([float(ms1[-1])]) if ms1 else None

def parse_scf_energy(text):
    """Converged SCF energy (hartree), taking the LAST value (with Stable=Opt
    the final stabilized SCF is what we want). Handles the PySCF printout
    ('converged SCF energy = ...') and Gaussian ('SCF Done:  E(UHF) = ...')."""
    ms = re.findall(r"converged SCF energy\s*=\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    if not ms:
        ms = re.findall(r"SCF Done:\s*E\([^)]*\)\s*=\s*"
                        r"(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    if not ms:
        return None
    return float(ms[-1].replace("D", "E").replace("d", "e"))

def parse_s2_annihilation(text):
    """(S**2 before, S**2 after) spin annihilation, from the LAST such line.
    Both Gaussian and PySCF print 'S**2 before annihilation X, after Y'."""
    ms = re.findall(r"S\*\*2 before annihilation\s+(-?\d+\.\d+)\s*,\s*"
                    r"after\s+(-?\d+\.\d+)", text)
    if not ms:
        return None
    b, a = ms[-1]
    return float(b), float(a)

def parse_mbs(text):
    """Parse every MBS quantity out of a Gaussian-format printout (works on
    both the Gaussian .out and the PySCF .minpop.out, since both use the same
    IOp(6/27=122) layout). Returns a dict of arrays (None where absent).
    Everything is taken from the LAST block (final Stable=Opt pass)."""
    labels, gross = parse_gross(text)
    s2 = parse_s2_annihilation(text)
    return {
        "escf": parse_scf_energy(text),                        # scalar hartree
        "s2_before": s2[0] if s2 else None,                    # scalar <S^2>
        "s2_after": s2[1] if s2 else None,                     # scalar <S^2>
        "gross_labels": labels,
        "gross": gross,                                        # [N, k] T,A,B,S
        "cond": parse_atomic_matrix(text, r"MBS Condensed to atoms"),
        "spin": parse_atomic_matrix(text, r"MBS Atomic-Atomic Spin Densities"),
        "musp": parse_mulliken_charges_spins(text),            # [N, 1 or 2]
        "musum": parse_mbs_charge_sum(text),                   # [2] charge, spin
        "da":   parse_density_matrix(text, r"Alpha\s+MBS Density Matrix"),
        "db":   parse_density_matrix(text, r"Beta\s+MBS Density Matrix"),
        "pt":   parse_density_matrix(text, r"Full MBS Mulliken population analysis"),
    }


# --------------------------------------------------------------------------- #
# small header parsers (the PySCF MinPop banner; Gaussian equivalents too)
# --------------------------------------------------------------------------- #
def _detect_code(text):
    if "converged SCF energy" in text or "pyscf" in text.lower():
        return "pyscf"
    if "Entering Gaussian System" in text or "SCF Done:" in text:
        return "gaussian"
    return "unknown"


def _parse_header(text):
    """charge / multiplicity / basis / natoms / nelectrons / molecule path.
    Handles the PySCF banner ('Charge: 0, Multiplicity: 1') and Gaussian
    ('Charge =  0 Multiplicity = 1')."""
    h = {"molecule": None, "charge": None, "multiplicity": None,
         "basis": None, "natoms": None, "nelectrons": None}
    m = re.search(r"^Molecule:\s*(\S+)", text, re.M)
    if m:
        h["molecule"] = m.group(1)
    m = re.search(r"Charge:\s*(-?\d+)\s*,\s*Multiplicity:\s*(\d+)", text)
    if not m:
        m = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", text)
    if m:
        h["charge"], h["multiplicity"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"^Basis:\s*(\S+)", text, re.M)
    if m:
        h["basis"] = m.group(1)
    m = re.search(r"Atoms:\s*(\d+)\s*,\s*Electrons:\s*(\d+)", text)
    if m:
        h["natoms"], h["nelectrons"] = int(m.group(1)), int(m.group(2))
    return h


def _parse_orbital_structure(text):
    """'UHF orbital structure: 22 alpha, 22 beta' -> (22, 22)."""
    m = re.search(r"orbital structure:\s*(\d+)\s*alpha\s*,\s*(\d+)\s*beta", text)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _parse_id(path):
    """rxn0081_point0_uhf_minpop.out -> ('rxn0081_point0', 'rxn0081', 0)."""
    stem = Path(path).stem
    m = re.search(r"(rxn(\d+))[_-]?point[_-]?(\d+)", stem, re.I)
    if not m:
        return stem, None, None
    return f"{m.group(1)}_point{int(m.group(3))}", m.group(1), int(m.group(3))


_SYM2Z = {s: z for z, s in _Z2SYM.items()}


def _read_orientation(text, header):
    """Read the LAST '<header> orientation:' block as [(Z, x, y, z), ...].

    Strict: returns None when the block is absent, and never falls back to the
    other orientation -- the PySCF MinPop printout has only a Standard orientation
    block, so a fallback would silently store standard coordinates under 'input'.
    """
    lines = text.splitlines()
    blocks = []
    for i, ln in enumerate(lines):
        if header in ln:
            j, atoms = i + 5, []          # skip the 4 header/rule lines
            while j < len(lines) and set(lines[j].strip()) != {"-"}:
                p = lines[j].split()
                if len(p) >= 6:
                    try:
                        atoms.append((int(p[1]), float(p[3]),
                                      float(p[4]), float(p[5])))
                    except ValueError:
                        pass
                j += 1
            if atoms:
                blocks.append(atoms)
    return blocks[-1] if blocks else None


def _read_xyz(path):
    """Read a plain .xyz -> [(Z, x, y, z), ...] (symbol or Z in column 1)."""
    try:
        lines = Path(path).read_text(errors="ignore").splitlines()
    except OSError:
        return None
    try:
        n = int(lines[0].split()[0])
    except (IndexError, ValueError):
        return None
    atoms = []
    for ln in lines[2:2 + n]:
        p = ln.split()
        if len(p) < 4:
            continue
        tok = p[0]
        z = _SYM2Z.get(tok.capitalize())
        if z is None:
            try:
                z = int(tok)
            except ValueError:
                return None
        try:
            atoms.append((z, float(p[1]), float(p[2]), float(p[3])))
        except ValueError:
            return None
    return atoms or None


def _resolve_input_xyz(molecule_path, source_out, xyz_dirs):
    """Find the input-orientation .xyz named on the 'Molecule:' line. Tries the
    literal path (it is usually a dead scratch path), then the basename next to
    the .out, then any --xyz-dir. Returns (atoms, where) or (None, None)."""
    if not molecule_path:
        return None, None
    cands = [Path(molecule_path)]
    base = Path(molecule_path).name
    if source_out:
        cands.append(Path(source_out).parent / base)
    for d in (xyz_dirs or []):
        cands.append(Path(d) / base)
    for c in cands:
        if c.is_file():
            atoms = _read_xyz(c)
            if atoms:
                return atoms, str(c)
    return None, None


# --------------------------------------------------------------------------- #
# array <-> json
# --------------------------------------------------------------------------- #
def _arr(a, units=None, columns=None, rows=None):
    """Wrap an array with its shape (and optional units/labels), or None."""
    if a is None:
        return None
    a = np.asarray(a, dtype=float)
    d = {"shape": list(a.shape), "data": a.tolist()}
    if units:
        d["units"] = units
    if columns:
        d["columns"] = list(columns)
    if rows:
        d["rows"] = list(rows)
    return d


def _unarr(d):
    if d is None:
        return None
    return np.asarray(d["data"], dtype=float).reshape(d["shape"])


# --------------------------------------------------------------------------- #
# build a record
# --------------------------------------------------------------------------- #
def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def build_record(path, xyz_dirs=None):
    """Parse one MinPop .out file (PySCF or Gaussian) into a JSON-ready dict."""
    path = Path(path)
    return build_record_from_text(path.read_text(errors="ignore"), source=path,
                                  xyz_dirs=xyz_dirs, origin="file")


def build_record_from_text(text, source=None, xyz_dirs=None,
                           origin="file"):
    """Build a record from the MinPop printout itself.

    This is what minpop_rohf.py / minpop_uhf.py call with their captured stdout,
    so a record written during a live run and a record parsed later from the
    saved .out come off the same parser and agree exactly (identical
    source.sha256, which is the hash of this text).

    Both orientations are stored. Gaussian outputs carry an Input orientation
    block; the PySCF MinPop printout does not, so the input geometry is read from
    the .xyz named on its 'Molecule:' line when that file is reachable (during a
    live run it always is). When it is not reachable, input_orientation is null
    rather than a copy of the standard frame.
    """
    hdr = _parse_header(text)
    mbs = parse_mbs(text)
    n_alpha, n_beta = _parse_orbital_structure(text)
    sysid, rxn, point = _parse_id(source) if source else (None, None, None)

    # Both orientations. Gaussian does not reorder atoms between the input and
    # standard frames, so the atom list is shared; we verify that and flag it.
    std = _read_orientation(text, "Standard orientation:")
    inp = _read_orientation(text, "Input orientation:")
    inp_src = "output_block" if inp else None
    if inp is None:
        # the PySCF MinPop printout has no Input orientation block; it names the
        # input .xyz on the 'Molecule:' line, so use that when we can reach it
        inp, where = _resolve_input_xyz(hdr["molecule"], source, xyz_dirs)
        inp_src = where

    ref = std or inp
    Z = [int(a[0]) for a in ref] if ref else None
    order_ok = None
    if std and inp:
        order_ok = [a[0] for a in std] == [a[0] for a in inp]

    def _coords(block):
        return ([[float(a[1]), float(a[2]), float(a[3])] for a in block]
                if block else None)

    natoms = hdr["natoms"] if hdr["natoms"] is not None else (
        len(Z) if Z is not None else None)

    rec = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "file": Path(source).name if source else None,
            "path": str(source) if source else None,
            "origin": origin,                  # "file" or "live_run"
            "sha256": _sha256_text(text),      # hash of the MinPop printout
            "code": _detect_code(text),
            "parsed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "system": {
            "id": sysid,
            "reaction": rxn,
            "point": point,
            "molecule": hdr["molecule"],
            "charge": hdr["charge"],
            "multiplicity": hdr["multiplicity"],
            "natoms": natoms,
            "nelectrons": hdr["nelectrons"],
        },
        "method": {
            "name": detect_method(text),
            "basis": hdr["basis"],
            "guess_mix": "Guess=Mix" in text,
            "stable_opt": "stability analysis" in text.lower(),
        },
        "geometry": {
            "units": "angstrom",
            "atomic_numbers": Z,
            "atomic_symbols": ([_Z2SYM.get(z, str(z)) for z in Z]
                               if Z is not None else None),
            "standard_orientation": _coords(std),
            "input_orientation": _coords(inp),
            "input_orientation_source": inp_src,
            "atom_order_consistent": order_ok,
        },
        "scf": {
            "energy_hartree": mbs["escf"],
            "s2_before_annihilation": mbs["s2_before"],
            "s2_after_annihilation": mbs["s2_after"],
            "n_alpha": n_alpha,
            "n_beta": n_beta,
        },
        "mbs": {
            "basis_function_labels": mbs["gross_labels"],
            "gross": _arr(mbs["gross"], units="electrons",
                          columns=["Total", "Alpha", "Beta", "Spin"]),
            "density_alpha": _arr(mbs["da"], units="electrons"),
            "density_beta": _arr(mbs["db"], units="electrons"),
            "population_full": _arr(mbs["pt"], units="electrons"),
            "condensed_to_atoms": _arr(mbs["cond"], units="electrons"),
            "atomic_spin_densities": _arr(mbs["spin"], units="electrons"),
            "mulliken": _arr(mbs["musp"], units="e",
                             columns=["charge", "spin"]),
            "mulliken_sum": (None if mbs["musum"] is None else
                             {"charge": float(mbs["musum"][0]),
                              "spin": (float(mbs["musum"][1])
                                       if len(mbs["musum"]) > 1 else None)}),
        },
    }
    # trim column labels to the columns actually present (ROHF prints no spin
    # column in some blocks, and the gross block can be Total-only)
    for _name in ("gross", "mulliken"):
        _blk = rec["mbs"][_name]
        if _blk is not None and len(_blk["shape"]) > 1:
            _blk["columns"] = _blk["columns"][:_blk["shape"][1]]
    return rec


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def export_json(text, json_path, source=None, xyz_dirs=None, indent=1):
    """One-call export used by minpop_rohf.py / minpop_uhf.py -json.

    `text` is the MinPop printout the driver just produced. Returns the path
    written. A '.gz' suffix on json_path compresses it (~8x on these matrices).
    """
    rec = build_record_from_text(text, source=source, xyz_dirs=xyz_dirs,
                                 origin="live_run" if source is None else "file")
    return write_record(rec, json_path, indent=indent)


def _open_w(path):
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "wb"), encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def _open_r(path):
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def write_record(rec, path, indent=1):
    """Write one record. A '.gz' suffix compresses (~10x on these matrices)."""
    with _open_w(path) as fh:
        json.dump(rec, fh, indent=indent)
        fh.write("\n")
    return Path(path)


def load_record(path, as_arrays=True):
    """Read a record back; matrices become numpy arrays when as_arrays."""
    with _open_r(path) as fh:
        rec = json.load(fh)
    if as_arrays:
        for k, v in rec.get("mbs", {}).items():
            if isinstance(v, dict) and "shape" in v:
                v["data"] = _unarr(v)
    return rec


def write_jsonl(recs, path):
    """One record per line -- the format ML tooling (pandas, HF datasets) likes."""
    n = 0
    with _open_w(path) as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return Path(path), n


def iter_jsonl(path, as_arrays=False):
    with _open_r(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if as_arrays:
                for k, v in rec.get("mbs", {}).items():
                    if isinstance(v, dict) and "shape" in v:
                        v["data"] = _unarr(v)
            yield rec


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# CSV export: long sparse elements + fixed-length ML features
#
# One table: a LONG (tidy, sparse) CSV with one row per nonzero element of
# every MBS quantity. This is the coordinate (COO) form of a sparse tensor:
# each row is an entry (system, quantity, i, j) -> value, so molecules of
# different sizes coexist because the AO/atom indices live in the rows, not
# the columns. Zeros are dropped, so the symmetry sparsity that bloats dense
# dumps disappears; pivot to a scipy.sparse matrix and SVD/UMAP work on it
# directly. Symmetric matrices are emitted as the upper triangle (i <= j,
# 1-based); mirror on load for a dense reconstruction.
# --------------------------------------------------------------------------- #

def parse_gross_atoms(text):
    """Gross block with the atom markers kept.

    Returns (ao_labels, atom_index_1based, elements, array[N, k]) or
    (None, None, None, None). The printout marks the atom only on its first AO;
    the marker carries forward over the atom's remaining AOs.
    """
    lines = text.splitlines()
    i = _find(lines, r"MBS Gross orbital populations")
    if i is None:
        return None, None, None, None
    j = i + 1
    if j < len(lines) and re.search(r"Total", lines[j]):
        j += 1
    labels, atoms, elems, rows = [], [], [], []
    cur_atom, cur_elem = None, None
    while j < len(lines):
        toks = lines[j].split()
        if not toks or not _INT.match(toks[0]):
            break
        if len(toks) >= 4 and _INT.match(toks[1]) and not _FLT.match(toks[2]):
            cur_atom, cur_elem, lbl = int(toks[1]), toks[2], toks[3]
            vals = _floats(toks[4:])
        elif len(toks) >= 2 and not _FLT.match(toks[1]):
            lbl, vals = toks[1], _floats(toks[2:])
        else:
            break
        if not vals:
            break
        labels.append(lbl)
        atoms.append(cur_atom)
        elems.append(cur_elem)
        rows.append(vals)
        j += 1
    if not rows:
        return None, None, None, None
    width = min(len(r) for r in rows)
    return labels, atoms, elems, np.array([r[:width] for r in rows])


def _parse_for_csv(text, source=None):
    """Everything both CSV writers need, parsed once."""
    hdr = _parse_header(text)
    mbs = parse_mbs(text)
    n_alpha, n_beta = _parse_orbital_structure(text)
    sysid, rxn, point = _parse_id(source) if source else (None, None, None)
    ao_labels, ao_atoms, ao_elems, gross = parse_gross_atoms(text)
    # element per ATOM comes from the AO->atom markers; the geometry block is
    # not required, so Gaussian and PySCF outputs parse alike
    atom_elems = {}
    if ao_atoms:
        for a, e in zip(ao_atoms, ao_elems):
            if a is not None and a not in atom_elems:
                atom_elems[a] = e
    return {
        "file": Path(source).name if source else None,
        "system_id": sysid, "reaction": rxn, "point": point,
        "method": detect_method(text), "basis": hdr["basis"],
        "charge": hdr["charge"], "multiplicity": hdr["multiplicity"],
        "natoms": hdr["natoms"] or (max(atom_elems) if atom_elems else None),
        "nelectrons": hdr["nelectrons"],
        "n_alpha": n_alpha, "n_beta": n_beta,
        "escf": mbs["escf"], "s2_before": mbs["s2_before"],
        "s2_after": mbs["s2_after"], "musum": mbs["musum"],
        "gross": gross, "gross_cols": ["total", "alpha", "beta", "spin"],
        "ao_labels": ao_labels, "ao_atoms": ao_atoms, "ao_elems": ao_elems,
        "atom_elems": atom_elems,
        "da": mbs["da"], "db": mbs["db"], "pt": mbs["pt"],
        "cond": mbs["cond"], "spinmat": mbs["spin"], "musp": mbs["musp"],
    }


_LONG_FIELDS = ["file", "system_id", "method", "quantity", "i", "j",
                "ao_i", "ao_j", "atom_i", "atom_j", "elem_i", "elem_j",
                "value"]


def long_rows(p, sparse_tol=0.0):
    """Yield tidy rows for one parsed run. abs(value) <= sparse_tol is dropped
    for vectors and matrices; scalars are always emitted."""
    base = {"file": p["file"], "system_id": p["system_id"],
            "method": p["method"]}
    def row(quantity, v, i="", j="", ao_i="", ao_j="",
            atom_i="", atom_j="", elem_i="", elem_j=""):
        return {**base, "quantity": quantity, "i": i, "j": j,
                "ao_i": ao_i, "ao_j": ao_j, "atom_i": atom_i, "atom_j": atom_j,
                "elem_i": elem_i, "elem_j": elem_j, "value": f"{v:.10g}"}

    for name, v in (("scf_energy_hartree", p["escf"]),
                    ("s2_before_annihilation", p["s2_before"]),
                    ("s2_after_annihilation", p["s2_after"])):
        if v is not None:
            yield row(name, v)

    labels, atoms, elems = p["ao_labels"], p["ao_atoms"], p["ao_elems"]
    if p["gross"] is not None and labels:
        for c in range(p["gross"].shape[1]):
            qn = "gross_" + p["gross_cols"][c]
            for i in range(p["gross"].shape[0]):
                v = p["gross"][i, c]
                if abs(v) > sparse_tol:
                    yield row(qn, v, i=i + 1, ao_i=labels[i],
                              atom_i=atoms[i], elem_i=elems[i])

    for qn, M in (("density_alpha", p["da"]), ("density_beta", p["db"]),
                  ("population_full", p["pt"])):
        if M is None or not labels:
            continue
        n = M.shape[0]
        for i in range(n):
            for j in range(i, n):                    # upper triangle
                v = M[i, j]
                if abs(v) > sparse_tol:
                    yield row(qn, v, i=i + 1, j=j + 1,
                              ao_i=labels[i], ao_j=labels[j],
                              atom_i=atoms[i], atom_j=atoms[j],
                              elem_i=elems[i], elem_j=elems[j])

    ae = p["atom_elems"]
    for qn, M in (("condensed_to_atoms", p["cond"]),
                  ("atomic_spin_densities", p["spinmat"])):
        if M is None:
            continue
        n = M.shape[0]
        for i in range(n):
            for j in range(i, n):
                v = M[i, j]
                if abs(v) > sparse_tol:
                    yield row(qn, v, i=i + 1, j=j + 1,
                              atom_i=i + 1, atom_j=j + 1,
                              elem_i=ae.get(i + 1, ""), elem_j=ae.get(j + 1, ""))

    if p["musp"] is not None:
        cols = ["mulliken_charge", "mulliken_spin"]
        for c in range(min(p["musp"].shape[1], 2)):
            for i in range(p["musp"].shape[0]):
                v = p["musp"][i, c]
                if abs(v) > sparse_tol:
                    yield row(cols[c], v, i=i + 1, atom_i=i + 1,
                              elem_i=ae.get(i + 1, ""))


def write_long_csv(parsed_runs, path, sparse_tol=0.0):
    import csv as _csv
    n = 0
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=_LONG_FIELDS)
        w.writeheader()
        for p in parsed_runs:
            for r in long_rows(p, sparse_tol=sparse_tol):
                w.writerow(r)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Serialize MinPop outputs (PySCF or Gaussian) to JSON "
                    "and/or ML-ready CSV.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--out", help="a single MinPop .out file")
    src.add_argument("--runs", help="directory tree to walk for MinPop outputs")
    ap.add_argument("--pattern", default="*minpop*.out",
                    help="glob for --runs mode (default: %(default)s)")
    ap.add_argument("--outdir", help="write one .json per input here "
                                     "(default: next to each input)")
    ap.add_argument("--jsonl", help="write ALL records to this JSON Lines file "
                                    "instead (use .jsonl.gz to compress)")
    ap.add_argument("--xyz-dir", action="append", metavar="DIR",
                    help="directory holding the input-orientation .xyz files "
                         "named on each output's 'Molecule:' line (repeatable).")
    ap.add_argument("--gzip", action="store_true",
                    help="gzip each per-file .json (.json.gz)")
    ap.add_argument("--indent", type=int, default=1,
                    help="json indent for per-file mode (0 = compact)")
    ap.add_argument("--csv-long", metavar="PATH",
                    help="write ALL runs' nonzero MBS elements to one tidy "
                         "sparse CSV (rows = elements; molecules of any size "
                         "coexist). Symmetric matrices: upper triangle, "
                         "1-based indices.")
    ap.add_argument("--sparse-tol", type=float, default=0.0,
                    help="drop |value| <= this from the long CSV "
                         "(default 0: drop exact zeros only)")
    args = ap.parse_args()

    if args.out:
        targets = [Path(args.out)]
    else:
        targets = sorted(p for p in Path(args.runs).rglob(args.pattern)
                         if not any(x.startswith(".") for x in p.parts))
    if not targets:
        sys.exit(f"[error] no MinPop outputs found "
                 f"(pattern {args.pattern!r} under {args.runs})")
    print(f"[info] {len(targets)} MinPop output(s)")

    want_csv = bool(args.csv_long)
    want_json = bool(args.jsonl or args.outdir) or not want_csv

    texts = []
    for t in targets:
        try:
            texts.append((t, t.read_text(errors="ignore")))
        except Exception as e:
            print(f"[warn] {t.name}: {type(e).__name__}: {e}")

    if want_json:
        recs, bad = [], 0
        for t, text in texts:
            try:
                rec = build_record_from_text(text, source=t,
                                             xyz_dirs=args.xyz_dir,
                                             origin="file")
            except Exception as e:
                print(f"[warn] {t.name}: {type(e).__name__}: {e}")
                bad += 1
                continue
            if rec["scf"]["energy_hartree"] is None:
                print(f"[warn] {t.name}: no SCF energy parsed")
            recs.append((t, rec))
        no_inp = sum(1 for _, r in recs
                     if r["geometry"]["input_orientation"] is None)
        if no_inp:
            print(f"[warn] {no_inp}/{len(recs)} record(s) have no input "
                  f"orientation (point --xyz-dir at the input .xyz files)")
        if args.jsonl:
            path, n = write_jsonl((r for _, r in recs), args.jsonl)
            print(f"[ok] wrote {n} record(s) -> {path} "
                  f"({path.stat().st_size/1e6:.1f} MB)")
        else:
            outdir = Path(args.outdir) if args.outdir else None
            if outdir:
                outdir.mkdir(parents=True, exist_ok=True)
            total = 0
            for t, rec in recs:
                suffix = ".json.gz" if args.gzip else ".json"
                dest = (outdir / (t.stem + suffix)) if outdir else \
                       t.with_name(t.stem + suffix)
                write_record(rec, dest, indent=(args.indent or None))
                total += dest.stat().st_size
            print(f"[ok] wrote {len(recs)} json file(s), "
                  f"{total/1e6:.1f} MB total")

    if want_csv:
        parsed = []
        for t, text in texts:
            try:
                parsed.append(_parse_for_csv(text, source=t))
            except Exception as e:
                print(f"[warn] {t.name}: {type(e).__name__}: {e}")
        if args.csv_long:
            n = write_long_csv(parsed, args.csv_long,
                               sparse_tol=args.sparse_tol)
            sz = Path(args.csv_long).stat().st_size / 1e6
            print(f"[ok] long CSV: {n} rows from {len(parsed)} run(s) -> "
                  f"{args.csv_long} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
