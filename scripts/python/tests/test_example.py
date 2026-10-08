"""
Regression check: rebuild example/ITPA_TAE_128_32.bin from example/standard2D.mat + example/NTP.mat
and compare every block with the file produced by the MATLAB scripts. Exits non-zero on failure.

    python verify_example.py            (run from scripts/preprocess/python)
    python verify_example.py /path/to/cuGMEC
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import cugmec_eq as ce

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
EX = ROOT / "example"

# Tolerance per block, relative to the block's own maximum (or to the value block for profile derivatives).
EXACT = 1e-12
JPB = 1e-6            # J_parallel is differentiated with an FFT instead of MATLAB's B-spline
UNKNOWN_FIT = {"Na", "Na_px"}  # the fit window and order used to make the example alpha density are not recorded

load = lambda name: {k: v for k, v in loadmat(EX / name).items() if not k.startswith("__")}
ref = ce.read_bin(EX / "ITPA_TAE_128_32.bin", 128, 32)
blocks, norm = ce.build_2d(load("standard2D.mat"), load("NTP.mat"), ion_mass=1.0,
                           ion_type=ce.MAXWELL, alpha_type=ce.MAXWELL, beam_type=ce.ABSENT)

failures = 0
for name, _ in ce.layout():
    r = ref[name][..., 0]
    m = np.asarray(blocks[name]).reshape(r.shape)
    base = name.split("_p")[0]
    scale = max(np.abs(r).max(), np.abs(ref[base]).max() if base in ref else 0.0)
    err = np.abs(r - m).max() / scale if scale > 0 else np.abs(m).max()
    tol = JPB if base == "JpB" else EXACT
    if name in UNKNOWN_FIT:
        print(f"  skip {name:14s} {err:.1e}  (fit settings of the example unknown)")
    elif err > tol:
        failures += 1
        print(f"  FAIL {name:14s} {err:.1e} > {tol:.0e}")

mat = loadmat(EX / "normalization2D.mat")
for name in ["B0", "L0", "VA0", "RHO0", "RHO1", "PSITMAX", "IonBeta", "BeamBeta", "gridNx", "gridNy", "NFP"]:
    a, b = float(np.ravel(mat[name])[0]), float(norm[name])
    if abs(a - b) > EXACT * max(abs(a), 1.0):
        failures += 1
        print(f"  FAIL {name}: MATLAB {a!r}, Python {b!r}")

ce.patch_param_h((ROOT / "src" / "cuGMEC_param.h").read_text(), norm)  # every constant must be patchable

print(f"{len(ce.layout())} blocks, {failures} failures")
sys.exit(1 if failures else 0)
