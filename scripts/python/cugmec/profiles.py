"""
Kinetic profiles from a file, converted to the NTP.mat form that build_2d() expects.

Files:
    .h5 / .hdf5   one coordinate dataset and the profile datasets, at the root or in a group /profiles
    .mat          NTP.mat as used by generateInput2D.m (rhoSample, neSample, ...); rhoSample is DESC rho
    other         text table: a header line with column names, then columns separated by spaces or commas;
                  the first column is the coordinate

Profiles, named as in NTP.mat: ne Te ni Ti Pi na Ta Pa nb Tb Pb
    n in 1e19 m^-3; T in keV for a Maxwellian species, or the critical speed v_c in m/s for a slowing-down
    species; P in Pa (needed for slowing-down species only).
Coordinates: rho (DESC rho), rho_tor, rho_pol, psi_n. All but rho need the g-file (see geqdsk.py).
"""
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from .equilibrium import ABSENT, KEV, MAXWELL, SLOWING

NAMES = ("ne", "Te", "ni", "Ti", "Pi", "na", "Ta", "Pa", "nb", "Tb", "Pb")
COORDS = ("rho", "rho_tor", "rho_pol", "psi_n")


def read_profiles(path):
    """(coordinate name, coordinate values, {profile name: values}) as stored in the file."""
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as f:
            g = f["profiles"] if "profiles" in f else f
            coords = [c for c in COORDS if c in g]
            if len(coords) != 1:
                raise ValueError(f"{path}: expected exactly one coordinate dataset out of {COORDS}, found {coords}")
            coord = coords[0]
            return coord, g[coord][()].ravel(), {n: g[n][()].ravel() for n in NAMES if n in g}
    if path.suffix.lower() == ".mat":
        m = loadmat(path)
        return "rho", np.ravel(m["rhoSample"]), {n: np.ravel(m[n + "Sample"]) for n in NAMES if n + "Sample" in m}
    with open(path) as f:
        header = f.readline()
    delimiter = "," if "," in header else None
    names = [h.strip() for h in header.strip().lstrip("#").split(delimiter)]
    data = np.loadtxt(path, skiprows=1, delimiter=delimiter, ndmin=2)
    if names[0] not in COORDS:
        raise ValueError(f"{path}: the first column must be one of {COORDS}, not {names[0]!r}")
    return names[0], data[:, 0], {n: data[:, i] for i, n in enumerate(names) if n in NAMES}


def required(types):
    """Profiles needed for species types (ion, alpha, beam) with codes MAXWELL, SLOWING, ABSENT."""
    need = ["ne", "Te"]
    for tag, kind in zip("iab", types):
        if kind != ABSENT:
            need += ["n" + tag, "T" + tag] + (["P" + tag] if kind == SLOWING else [])
    return need


def to_ntp(rho, prof, types):
    """NTP.mat-style dict on DESC rho, sorted, without points outside the DESC boundary (rho > 1)."""
    missing = [n for n in required(types) if n not in prof]
    if missing:
        raise ValueError(f"profiles missing for the chosen species types: {missing}")
    keep = rho <= 1.0 + 1e-12
    order = np.argsort(rho[keep])
    out = {"rhoSample": rho[keep][order][None, :]}
    for name, values in prof.items():
        out[name + "Sample"] = np.asarray(values, dtype=float)[keep][order][None, :]
    return out


def kinetic_pressure(ntp, types):
    """Total pressure p(rho) [Pa] = electrons + every species present, from an NTP-style dict."""
    rho = ntp["rhoSample"].ravel()
    s = lambda n: ntp[n + "Sample"].ravel()
    p = 1e19 * s("ne") * s("Te") * KEV
    for tag, kind in zip("iab", types):
        if kind == MAXWELL:
            p = p + 1e19 * s("n" + tag) * s("T" + tag) * KEV
        elif kind == SLOWING:
            p = p + s("P" + tag)
    return lambda r: np.interp(r, rho, p)
