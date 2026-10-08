"""
Output conversion on a synthetic run directory: every file holds its own flat index, so each decoded value can be checked.

    python -m pytest tests          or          python tests/test_outputs.py
"""
import shutil
from pathlib import Path

import h5py
import numpy as np

from cugmec import params
from cugmec.outputs import CONTINUE_FIELDS, convert_outputs

ROOT = Path(__file__).resolve().parents[3]
EX = ROOT / "example"
SWITCHES = ["ifDiagAmplitude", "ifDiagDiffusivity", "ifOutputPhi", "ifOutputA", "ifOutputPhaseSpaceDeltaF",
            "ifOutputPhaseSpacePower", "ifOutputPhaseSpaceF0", "ifOutputPitchSpaceDeltaF"]


def _write(path, shape, dtype):
    data = np.arange(int(np.prod(shape))).astype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(path)


def _run(tmp):
    text = params.patch_param_h((EX / "cuGMEC_param.h").read_text(), {
        "totalSteps": 20, "diagSteps": 5, "outputSteps": 10, "continueSteps": 0, "devNums": 2, "ppcNums": 1,
        "gridE": 8, "gridPphi": 6, "gridLambda": 4, "gridVpara": 6, "gridVperp": 4, "ifDiagShearing": False,
        **{k: True for k in SWITCHES}})
    out = tmp / "output"
    (out / "final").mkdir(parents=True)
    (out / "final" / "cuGMEC_param.h").write_text(text)
    shutil.copy(EX / "ITPA_TAE_128_32.bin", out / "final" / "ITPA_TAE_128_32.bin")
    p = params.parse_param_h(text)
    nx, ny, nz, nd, no, nm = 128, 32, 16, 5, 3, 1
    _write(out / "final/amplitude.bin", (nd, nx, nm), "<f8")
    _write(out / "final/RealMode.bin", (nd, nx, nm), "<f8")
    _write(out / "final/ImagMode.bin", (nd, nx, nm), "<f8")
    _write(out / "final/IonDiffusivity.bin", (nd, nm, nx), "<f8")
    _write(out / "final/AlphaDiffusivity.bin", (nd, nm, nx), "<f8")
    _write(out / "final/totalPhi.bin", (no, ny, nx, nz), "<f8")
    _write(out / "final/totalA.bin", (no, ny, nx, nz + 1), "<f8")  # wrong size on purpose
    _write(out / "final/shearing.bin", (nd, nx), "<f8")            # switch off on purpose
    for name in ("Phi", "w"):
        _write(out / f"final/{name}.bin", (ny, nx, nz), "<f8")
    for s in ("Ion", "Alpha"):
        _write(out / f"initial/{s}Density.bin", (nx,), "<f4")
        _write(out / f"final/{s}PhaseDeltaF.bin", (no, 8, 6, 4), "<f8")
        _write(out / f"final/{s}PhasePower.bin", (no, nm, 8, 6, 4), "<f8")
        _write(out / f"initial/{s}PhaseSpaceF0.bin", (8, 6, 4), "<f8")
        _write(out / f"final/{s}PitchDeltaF.bin", (no, 6, 4), "<f8")
    _write(out / "final/MHDContinue_20.bin", (len(CONTINUE_FIELDS), ny, nx, nz), "<f8")

    # PICContinue: per species {const, keys[picHost], values[devNums][7][picDev]}; value = component * 1e6 + marker
    pic_dev = nx * ny * nz // 2
    with open(out / "final/PICContinue_20.bin", "wb") as f:
        for const in (0.5, 0.25):
            np.array([const], "<f4").tofile(f)
            np.arange(2 * pic_dev, dtype="<i4").tofile(f)
            marker = np.arange(2 * pic_dev).reshape(2, 1, pic_dev)
            (np.arange(7).reshape(1, 7, 1) * 1e6 + marker).astype("<f4").tofile(f)
    return out, p


def test_outputs(tmp_path):
    out, p = _run(tmp_path)
    report = convert_outputs(out, tmp_path / "run.h5", restart=True)
    text = "\n".join(report)
    assert "ignored  final/shearing.bin" in text
    assert "skipped  final/totalA.bin" in text
    with h5py.File(tmp_path / "run.h5", "r") as h:
        phi = h["fields/Phi"]
        assert phi.shape == (3, 32, 128, 16) and phi.attrs["dims"] == "time_output,y,x,z"
        assert phi[2, 5, 7, 3] == ((2 * 32 + 5) * 128 + 7) * 16 + 3
        assert h["diagnostics/IonDiffusivity"].shape == (5, 1, 128)
        assert h["diagnostics/amplitude"][4, 100, 0] == 4 * 128 + 100
        assert list(h["time/t_diag"][:]) == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert h["grid"].attrs["diag_y_index"] == 16  # devNums = 2: first row of device 1 (devNy = 16)
        assert h["phase_space/Ion/power"].shape == (3, 1, 8, 6, 4)
        e, pphi, lam = (h[f"phase_space/Ion/{k}"][:] for k in ("E", "Pphi", "Lambda"))
        assert e.size == 8 and pphi.size == 6 and lam.size == 4 and np.all(np.diff(pphi) > 0)
        assert h["pitch_space/Ion/vpara"][0] == -p["IonVmax"]
        assert h["initial/Ion/density"].dtype == np.float32
        assert h["equilibrium/blocks/B"].shape == (128, 32, 1)
        assert h["param"].attrs["gridNx"] == 128
        x, vpara = h["restart/final/Alpha/x"][:], h["restart/final/Alpha/vpara"][:]
        assert np.array_equal(x, np.arange(2 * 32768, dtype=np.float32))
        assert np.array_equal(vpara, 3e6 + np.arange(2 * 32768, dtype=np.float32))
        assert h["restart/final/Alpha/const"][0] == 0.25


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_outputs(Path(tmp))
    print("ok  test_outputs")
