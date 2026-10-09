"""
gEQDSK reader, flux maps, kinetic profiles, and the DESC evaluation port (skipped without DESC).

    python -m pytest tests
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import loadmat, savemat

from cugmec import profiles
from cugmec.geqdsk import GEqdsk, arc_length_theta, read_geqdsk

ROOT = Path(__file__).resolve().parents[3]
R0, A, Q, PSI_B = 3.0, 1.0, 2.0, 0.4  # concentric circles with psi = PSI_B (r/a)^2 and constant q


def _fmt(values):
    out, line = [], ""
    for i, v in enumerate(values, 1):
        line += f"{v:16.9E}"
        if i % 5 == 0:
            out.append(line)
            line = ""
    return out + ([line] if line else [])


def write_geqdsk(path, nw=65, nh=65, sign=1.0):
    """Synthetic g-file. sign=-1 stores psirz with the opposite sign to simag and sibry, as some writers do."""
    R = np.linspace(R0 - 1.5 * A, R0 + 1.5 * A, nw)
    Z = np.linspace(-1.5 * A, 1.5 * A, nh)
    psi = PSI_B * ((R[None, :] - R0) ** 2 + Z[:, None] ** 2) / A**2
    psin = np.linspace(0, 1, nw)
    t = np.linspace(0, 2 * np.pi, 41)[:-1]
    lines = [f"{'synthetic':48s}{0:4d}{nw:4d}{nh:4d}"]
    lines += _fmt([3 * A, 3 * A, R0, R0 - 1.5 * A, 0.0, R0, 0.0, 0.0, PSI_B, 2.0,
                   1e6, 0.0, 0.0, R0, 0.0, 0.0, 0.0, PSI_B, 0.0, 0.0])
    lines += _fmt(np.full(nw, 6.0))                 # fpol
    lines += _fmt(1e4 * (1 - psin))                 # pres
    lines += _fmt(np.zeros(nw))                     # ffprim
    lines += _fmt(np.full(nw, -1e4 / PSI_B))        # pprime
    lines += _fmt(sign * psi.ravel())               # psirz
    lines += _fmt(np.full(nw, -Q))                  # qpsi (negative: only |q| is used)
    lines += [f"{t.size:5d}{4:5d}"]
    lines += _fmt(np.column_stack([R0 + A * np.cos(t), A * np.sin(t)]).ravel())
    lines += _fmt([R0 - 1.4, -1.4, R0 + 1.4, -1.4, R0 + 1.4, 1.4, R0 - 1.4, 1.4])
    Path(path).write_text("\n".join(lines) + "\n")


def test_reader_and_maps(tmp_path):
    write_geqdsk(tmp_path / "g.eqdsk")
    d = read_geqdsk(tmp_path / "g.eqdsk")
    assert d["nw"] == 65 and d["psirz"].shape == (65, 65) and d["rbbbs"].size == 40 and d["rlim"].size == 4
    g = GEqdsk(tmp_path / "g.eqdsk")
    assert not g.flipped
    assert abs(g.q(0.3) - Q) < 1e-12 and abs(g.pressure(0.25) - 7500) < 1e-6
    assert abs(float(g.pressure_from_pprime(0.25)) - 7500) < 1e-6         # integral of pprime agrees with pres
    assert abs(g.psit(1.0) - Q * PSI_B) < 1e-12                       # psi_t = integral of q dpsi_p
    assert np.allclose(g.rho(np.array([0.25, 0.81]), 0.81), [0.25**0.5 / 0.9, 1.0])
    assert np.allclose(g.psin_of_rho(np.array([0.5]), 0.81), 0.25 * 0.81, rtol=1e-6)
    assert np.allclose(g.to_psin("rho_pol", [0.5]), 0.25) and np.allclose(g.to_psin("rho_tor", [0.5]), 0.25, rtol=1e-6)
    R, Z = g.trace(0.64, 128)
    assert np.allclose(np.hypot(R - R0, Z), 0.8 * A, atol=1e-6)        # circle of radius a sqrt(psi_n)
    theta = arc_length_theta(R, Z)
    assert theta[0] == 0 and np.allclose(np.diff(theta), 2 * np.pi / 128, rtol=1e-3)


def test_flipped_psi_sign(tmp_path):
    write_geqdsk(tmp_path / "g.eqdsk", sign=-1.0)
    g = GEqdsk(tmp_path / "g.eqdsk")
    assert g.flipped and abs(float(g.psin(R0 + 0.5, 0.0)) - 0.25) < 1e-9
    assert abs(float(g.pressure_from_pprime(0.0)) - 1e4) < 1e-6


def test_profiles_formats(tmp_path):
    x = np.linspace(0, 1, 11)
    cols = {"ne": 5 - 4 * x, "Te": 2 - x, "ni": 4 - 3 * x, "Ti": 1.5 - x}
    np.savetxt(tmp_path / "p.txt", np.column_stack([x] + list(cols.values())), header="psi_n " + " ".join(cols), comments="")
    with h5py.File(tmp_path / "p.h5", "w") as f:
        f["profiles/rho_pol"] = np.sqrt(x)
        for k, v in cols.items():
            f[f"profiles/{k}"] = v
    savemat(tmp_path / "NTP.mat", {"rhoSample": x[None, :], **{k + "Sample": v[None, :] for k, v in cols.items()}})
    for name, coord in (("p.txt", "psi_n"), ("p.h5", "rho_pol"), ("NTP.mat", "rho")):
        c, xs, prof = profiles.read_profiles(tmp_path / name)
        assert c == coord and set(prof) == set(cols) and np.allclose(prof["Te"], cols["Te"])
    ntp = profiles.to_ntp(x, cols, (1, 3, 3))
    assert ntp["rhoSample"].shape == (1, 11) and np.allclose(ntp["TiSample"].ravel(), cols["Ti"])
    p = profiles.kinetic_pressure(ntp, (1, 3, 3))(0.0)
    assert abs(p - 1e19 * (5 * 2 + 4 * 1.5) * 1.6021766208e-16) < 1e-6
    with pytest.raises(ValueError, match="na"):
        profiles.to_ntp(x, cols, (1, 1, 3))


def test_desc_eval_matches_notebook(tmp_path):
    """cugmec.desc_eval.standard2d gives exactly what the notebook's outputStandard2D writes to standard2D.mat."""
    desc = pytest.importorskip("desc")
    import desc.examples

    from cugmec import desc_eval

    eq = desc.examples.get("DSHAPE")
    nb = json.loads((ROOT / "scripts/equilibrium/output2D_0170.ipynb").read_text())
    ns = {"np": np, "desc": desc, "savemat": savemat, "eq": eq, "outputPath": str(tmp_path)}
    exec("".join(nb["cells"][5]["source"]), ns)
    rho, theta = ns["getRhoThetaPEST"](16, 0.1, 0.9, 8)
    ns["outputStandard2D"](rho, theta, 2)
    ref = {k: v for k, v in loadmat(tmp_path / "standard2D.mat").items() if not k.startswith("__")}
    mine = desc_eval.standard2d(eq, 0.1, 0.9, 16, 8, 2)
    assert set(ref) == set(mine)
    for k in ref:
        assert np.array_equal(np.asarray(mine[k], float).reshape(np.shape(ref[k])), ref[k]), k
