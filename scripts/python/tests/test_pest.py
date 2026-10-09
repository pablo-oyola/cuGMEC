"""
Direct gEQDSK path (pyCOCOS) on the synthetic g-file of test_geqdsk.py, where everything is analytic: concentric
circular surfaces psi = PSI_B (r/a)^2 around R0, constant F, and p linear in psi. Skipped without pyCOCOS.

    q(r)      = F a^2 / (2 PSI_B sqrt(R0^2 - r^2))                      (flux-map q, not the file's qpsi)
    theta*    = 2 atan( sqrt((R0 - r)/(R0 + r)) tan(alpha/2) )           (PEST angle, counter-clockwise)
    |B|^2     = (F^2 + (2 PSI_B r / a^2)^2) / R^2
    J_par     = SIGN F p'(psi) / B
"""
import h5py
import numpy as np
import pytest

from test_geqdsk import A, PSI_B, R0, write_geqdsk

pytest.importorskip("pycocos")

from cugmec import cli  # noqa: E402
from cugmec.geqdsk import GEqdsk  # noqa: E402
from cugmec.pest import SIGN, PestMap  # noqa: E402

F = 6.0


def test_pest_map_is_analytic(tmp_path):
    write_geqdsk(tmp_path / "g.eqdsk")
    g = GEqdsk(tmp_path / "g.eqdsk")
    pm = PestMap(g, psin_b=0.81)
    r = A * np.sqrt(pm.psin_s)
    q = F * A**2 / (2 * PSI_B * np.sqrt(R0**2 - r**2))
    assert np.max(np.abs(pm.q_s - q) / q) < 1e-5

    S, rep = pm.standard2d(0.1, 0.9, 32, 16)
    R, Z = np.asarray(S["R"]), np.asarray(S["Z"])
    rr = np.hypot(R - R0, Z)
    assert np.max(np.abs(g.psin(R, Z) - pm.psin_of_rho(np.asarray(S["rho"])))) < 1e-5   # right flux surfaces
    alpha = np.arctan2(Z, R - R0)
    theta_ccw = 2 * np.arctan(np.sqrt((R0 - rr) / (R0 + rr)) * np.tan(alpha / 2))
    assert np.max(np.abs(np.exp(1j * theta_ccw) - np.exp(-1j * np.asarray(S["theta_pest"])))) < 1e-5   # DESC orientation
    B = np.sqrt(F**2 + (2 * PSI_B * rr / A**2) ** 2) / R
    assert np.max(np.abs(np.asarray(S["B"]) - B) / B) < 1e-5
    assert np.max(np.abs(np.asarray(S["J_parallel"]) - SIGN * F * (-1e4 / PSI_B) / B) / (1e4 / PSI_B * F / B)) < 1e-3
    assert np.all(np.asarray(S["JPEST"]) > 0) and rep["pest_jacobian_max_rel_error"] < 1e-4


def test_from_geqdsk_end_to_end(tmp_path):
    write_geqdsk(tmp_path / "g.eqdsk")
    x = np.linspace(0, 1, 21)
    np.savetxt(tmp_path / "p.txt", np.column_stack([x, 5 - 4 * x, 2 - x, 4 - 3 * x, 1.5 - x]),
               header="psi_n ne Te ni Ti", comments="")
    cli.main(["from-geqdsk", str(tmp_path / "g.eqdsk"), str(tmp_path / "p.txt"), "-o", str(tmp_path / "eq.h5"),
              "--ion-mass", "2", "--types", "1", "3", "3", "--psin-b", "0.81", "--grid", "128", "16",
              "--case", str(tmp_path / "case"), "--template", str(cli.Path(__file__).resolve().parents[3] / "example/cuGMEC_param.h"),
              "--set", "ifAlpha=false", "--set", "ifBeam=false"])
    with h5py.File(tmp_path / "eq.h5", "r") as f:
        assert f["blocks/B"].shape == (128, 16, 1) and f["inputs"].attrs["method"] == "pycocos"
        assert abs(f["normalization/B0"][()] - F / R0) < 1e-12
    assert (tmp_path / "case" / "MHDEquilibrium_128_16.bin").exists()
