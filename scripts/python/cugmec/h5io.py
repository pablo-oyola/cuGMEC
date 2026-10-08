"""
Equilibrium input in HDF5. The binary cuGMEC reads is generated from this file.

Layout (format_version 1):

    /                      attrs: format, format_version, created, generator
    /grid                  attrs: gridNx, gridNy, equilNz, NFP, ghost
        x (Nx)             radial coordinate in [0, 1]
        rho (Nx)           rho_tor = RHO0 + x (RHO1 - RHO0)
        y (Ny)             theta_PEST cell centres in [-pi, pi)
        y_ghost (Ny + 4)   y extended by the ghost cells used by the SFA blocks
    /normalization/<name>  scalars B0, L0, VA0, RHO0, RHO1, PSITMAX, IonBeta, AlphaBeta, BeamBeta; attrs unit, description
    /blocks/<name>         (Nx, ny, equilNz) float64 in cuGMEC's normalized form; attrs description, normalization, index
    /inputs                how the blocks were made: species types, ion mass, fit settings, source files and checksums
    /inputs/profiles/      the profile samples (NTP.mat), when available
"""
import datetime
import hashlib
import json

import h5py
import numpy as np

from . import __version__
from .equilibrium import GHOST, describe_block, layout, read_bin, write_bin

FORMAT = "cuGMEC equilibrium"
FORMAT_VERSION = 1

NORMALIZATION = {
    "B0": ("T", "|B| on the magnetic axis"),
    "L0": ("m", "major radius of the magnetic axis"),
    "VA0": ("m/s", "B0 / sqrt(mu0 IonMass m_p n_i), with n_i at the first profile sample"),
    "RHO0": ("1", "rho_tor of the first radial grid point"),
    "RHO1": ("1", "rho_tor of the last radial grid point"),
    "PSITMAX": ("Wb/rad", "toroidal flux at the boundary / 2 pi"),
    "IonBeta": ("1", "thermal ion pressure / (B0^2 / 2 mu0) at rho = RHO0"),
    "AlphaBeta": ("1", "alpha-slot pressure / (B0^2 / 2 mu0) at rho = RHO0"),
    "BeamBeta": ("1", "beam-slot pressure / (B0^2 / 2 mu0) at rho = RHO0"),
}
GRID_KEYS = ("gridNx", "gridNy", "NFP")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_equilibrium(path, blocks, norm, equilNz=1, inputs=None, profiles=None):
    """
    blocks: {name: (Nx, ny) or (Nx, ny, equilNz) array} for all 159 blocks.
    norm:   the normalization and grid values (as returned by build_2d).
    inputs: optional flat dict of provenance attributes; profiles: optional {name: array}.
    """
    with h5py.File(path, "w") as f:
        stamp(f, FORMAT, FORMAT_VERSION)
        write_equilibrium(f, blocks, norm, equilNz, inputs, profiles)


def stamp(f, fmt, version):
    f.attrs.update(format=fmt, format_version=version, generator=f"cugmec {__version__}",
                   created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))


def write_equilibrium(f, blocks, norm, equilNz=1, inputs=None, profiles=None):
    """Write the grid, normalization, blocks and inputs groups into an open HDF5 file or group."""
    Nx, Ny, NFP = (int(norm[k]) for k in GRID_KEYS)
    g = f.create_group("grid")
    g.attrs.update(gridNx=Nx, gridNy=Ny, equilNz=equilNz, NFP=NFP, ghost=GHOST)
    x = np.linspace(0.0, 1.0, Nx)
    g["x"] = x
    g["rho"] = norm["RHO0"] + x * (norm["RHO1"] - norm["RHO0"])
    g["y"] = -np.pi + (np.arange(Ny) + 0.5) * 2 * np.pi / Ny
    g["y_ghost"] = -np.pi + (np.arange(Ny + 2 * GHOST) - (GHOST - 0.5)) * 2 * np.pi / Ny

    n = f.create_group("normalization")
    for name, (unit, text) in NORMALIZATION.items():
        n[name] = float(norm[name])
        n[name].attrs.update(unit=unit, description=text)

    b = f.create_group("blocks")
    for index, (name, ghosted) in enumerate(layout()):
        ny = Ny + 2 * GHOST if ghosted else Ny
        a = np.asarray(blocks[name], dtype=np.float64)
        a = np.broadcast_to(a[:, :, None] if a.shape == (Nx, ny) else a, (Nx, ny, equilNz))
        d = b.create_dataset(name, data=a)
        description, normalization = describe_block(name)
        d.attrs.update(description=description, normalization=normalization, index=index, ghosted=ghosted)

    i = f.create_group("inputs")
    for key, value in (inputs or {}).items():
        i.attrs[key] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
    if profiles:
        p = i.create_group("profiles")
        for name, value in profiles.items():
            p[name] = np.ravel(value)


def load_equilibrium(path):
    """(blocks, norm, info): blocks as (Nx, ny, equilNz) arrays, norm as in build_2d, info with grid and input attrs."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        if int(f.attrs["format_version"]) > FORMAT_VERSION:
            raise ValueError(f"{path} has format_version {f.attrs['format_version']}, this tool reads up to {FORMAT_VERSION}")
        grid = {k: int(v) for k, v in f["grid"].attrs.items()}
        norm = {name: float(f["normalization"][name][()]) for name in f["normalization"]}
        norm.update(gridNx=grid["gridNx"], gridNy=grid["gridNy"], NFP=grid["NFP"])
        blocks = {name: f["blocks"][name][()] for name, _ in layout()}
        info = {"grid": grid, "inputs": {k: _attr(v) for k, v in f["inputs"].attrs.items()}}
    return blocks, norm, info


def export_bin(h5_path, bin_path):
    """Write the binary cuGMEC reads from an equilibrium HDF5 file. Returns (norm, info)."""
    blocks, norm, info = load_equilibrium(h5_path)
    write_bin(bin_path, blocks, norm["gridNx"], norm["gridNy"], info["grid"]["equilNz"])
    return norm, info


def import_bin(bin_path, h5_path, Nx, Ny, equilNz=1, norm=None, NFP=1):
    """Wrap an existing equilibrium binary in HDF5. norm (e.g. from cuGMEC_param.h) fills /normalization."""
    blocks = read_bin(bin_path, Nx, Ny, equilNz)
    values = {name: float("nan") for name in NORMALIZATION}
    values.update({k: v for k, v in (norm or {}).items() if k in NORMALIZATION})
    values.update(gridNx=Nx, gridNy=Ny, NFP=NFP)
    save_equilibrium(h5_path, blocks, values, equilNz,
                     inputs={"imported_from": str(bin_path), "imported_sha256": sha256(bin_path)})


def _attr(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            pass
    return v.item() if isinstance(v, np.generic) else v
