"""
Checks against the shipped example (example/ITPA_TAE_128_32.bin, made by the MATLAB scripts).

    python -m pytest tests          or          python tests/test_example.py
"""
import filecmp
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from cugmec import cli, h5io, params
from cugmec import equilibrium as ce

ROOT = Path(__file__).resolve().parents[3]
EX = ROOT / "example"

EXACT = 1e-12          # relative to the block's maximum (or to the value block for derivatives)
JPB = 1e-6             # J_parallel is differentiated with an FFT instead of MATLAB's B-spline
UNKNOWN_FIT = {"Na", "Na_px"}  # the fit settings used to make the example alpha density are not recorded


def _load(name):
    return {k: v for k, v in loadmat(EX / name).items() if not k.startswith("__")}


def _build():
    return ce.build_2d(_load("standard2D.mat"), _load("NTP.mat"), ion_mass=1.0,
                       ion_type=ce.MAXWELL, alpha_type=ce.MAXWELL, beam_type=ce.ABSENT)


def test_blocks_match_matlab():
    ref = ce.read_bin(EX / "ITPA_TAE_128_32.bin", 128, 32)
    blocks, _ = _build()
    failures = []
    for name, _ in ce.layout():
        if name in UNKNOWN_FIT:
            continue
        r = ref[name][..., 0]
        m = np.asarray(blocks[name]).reshape(r.shape)
        base = name.split("_p")[0]
        scale = max(np.abs(r).max(), np.abs(ref[base]).max() if base in ref else 0.0)
        err = np.abs(r - m).max() / scale if scale > 0 else np.abs(m).max()
        if err > (JPB if base == "JpB" else EXACT):
            failures.append(f"{name}: {err:.1e}")
    assert not failures, failures


def test_normalization_matches_matlab():
    _, norm = _build()
    mat = loadmat(EX / "normalization2D.mat")
    for name in ["B0", "L0", "VA0", "RHO0", "RHO1", "PSITMAX", "IonBeta", "BeamBeta", "gridNx", "gridNy", "NFP"]:
        a = float(np.ravel(mat[name])[0])
        assert abs(a - float(norm[name])) <= EXACT * max(abs(a), 1.0), name


def test_hdf5_roundtrip(tmp_path):
    blocks, norm = _build()
    ce.write_bin(tmp_path / "direct.bin", blocks, 128, 32)
    h5io.save_equilibrium(tmp_path / "eq.h5", blocks, norm, inputs={"ion_mass": 1.0, "fit": {"ne": [512, 12]}})
    h5io.export_bin(tmp_path / "eq.h5", tmp_path / "via_h5.bin")
    assert filecmp.cmp(tmp_path / "direct.bin", tmp_path / "via_h5.bin", shallow=False)
    _, norm2, info = h5io.load_equilibrium(tmp_path / "eq.h5")
    assert norm2 == {k: norm[k] for k in norm2}
    assert info["inputs"]["fit"] == {"ne": [512, 12]}


def test_case_directory(tmp_path):
    cli.main(["prepare", str(EX / "standard2D.mat"), str(EX / "NTP.mat"), "-o", str(tmp_path / "eq.h5"),
              "--ion-mass", "1.0", "--types", "1", "1", "3"])
    case = tmp_path / "itpa"
    cli.main(["case", str(tmp_path / "eq.h5"), str(case), "--template", str(EX / "cuGMEC_param.h"), "--set", "dt=0.025"])
    p = params.parse_param_h((case / "cuGMEC_param.h").read_text())
    assert p["inputDir"] == str(case) and p["outputDir"] == str(case / "output")
    assert p["MHDEquilibrium"] == "MHDEquilibrium_128_32.bin" and p["dt"] == 0.025 and p["IonMass"] == 1.0
    assert abs(p["VA0"] / 1.459946004856486e+07 - 1) < 1e-14
    blocks = ce.read_bin(case / p["MHDEquilibrium"], 128, 32)
    ref = ce.read_bin(EX / "ITPA_TAE_128_32.bin", 128, 32)
    assert np.allclose(blocks["SFAconxy_py"], ref["SFAconxy_py"], rtol=0, atol=1e-14)


if __name__ == "__main__":
    import tempfile
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as tmp:
                test(*([Path(tmp)] if test.__code__.co_argcount else []))
            print(f"ok  {name}")
