"""
Read a gEQDSK file (EFIT g-file) and map between its flux coordinates.

Only magnitudes are used (|q|, |F|, |psi_boundary - psi_axis|), so the sign conventions of the file's COCOS do not
matter. The COCOS index only decides whether psi is per radian (COCOS 1-8) or the total flux (COCOS 11-18).

Coordinates:
    psi_n    (psi - psi_axis) / (psi_boundary - psi_axis) of the g-file
    rho_pol  sqrt(psi_n)
    rho_tor  sqrt(psi_t / psi_t(psi_n = 1)), normalized to the g-file boundary
    rho      DESC's sqrt(psi_t / psi_t(psi_n = psin_b)), normalized to the surface used as the DESC boundary
"""
import re

import numpy as np
from scipy.interpolate import PchipInterpolator, RectBivariateSpline
from scipy.optimize import brentq

_FLOAT = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[eEdD][+-]?\d+)?")
_SCALARS = ("rdim", "zdim", "rcentr", "rleft", "zmid", "rmaxis", "zmaxis", "simag", "sibry", "bcentr",
            "current", None, None, None, None, None, None, None, None, None)


def read_geqdsk(path):
    """Raw contents of a gEQDSK file as a dict (psirz has shape (nh, nw), rows along Z)."""
    with open(path) as f:
        lines = f.read().splitlines()
    header = lines[0]
    nw, nh = (int(t) for t in header.split()[-2:])
    need = 20 + 4 * nw + nw * nh + nw
    values, i = [], 1
    while len(values) < need:
        values += [float(x.replace("D", "E").replace("d", "e")) for x in _FLOAT.findall(lines[i])]
        i += 1
    if len(values) != need:
        raise ValueError(f"{path}: unexpected number of values before the boundary section")
    while not lines[i].strip():
        i += 1
    nbbbs, limitr = (int(t) for t in lines[i].split()[:2])
    rest = []
    for line in lines[i + 1:]:
        if len(rest) >= 2 * (nbbbs + limitr):
            break
        rest += [float(x.replace("D", "E").replace("d", "e")) for x in _FLOAT.findall(line)]

    d = {"header": header[:48].strip(), "nw": nw, "nh": nh}
    d.update({name: v for name, v in zip(_SCALARS, values[:20]) if name})
    off = 20
    for name in ("fpol", "pres", "ffprim", "pprime"):
        d[name] = np.array(values[off:off + nw])
        off += nw
    d["psirz"] = np.array(values[off:off + nw * nh]).reshape(nh, nw)
    off += nw * nh
    d["qpsi"] = np.array(values[off:off + nw])
    bbbs = np.array(rest[:2 * nbbbs]).reshape(-1, 2)
    lim = np.array(rest[2 * nbbbs:2 * (nbbbs + limitr)]).reshape(-1, 2)
    d["rbbbs"], d["zbbbs"] = bbbs[:, 0], bbbs[:, 1]
    d["rlim"], d["zlim"] = lim[:, 0], lim[:, 1]
    return d


class GEqdsk:
    """A g-file with splines for psi_n(R, Z) and the 1D profiles, and the flux-coordinate maps."""

    def __init__(self, path, cocos=1):
        self.path, self.cocos = str(path), cocos
        self.data = d = read_geqdsk(path)
        self.R = d["rleft"] + d["rdim"] * np.linspace(0.0, 1.0, d["nw"])
        self.Z = d["zmid"] - d["zdim"] / 2 + d["zdim"] * np.linspace(0.0, 1.0, d["nh"])
        self.axis = (d["rmaxis"], d["zmaxis"])
        self._psi = RectBivariateSpline(self.Z, self.R, d["psirz"])
        # psirz must grow from the axis to the boundary in the same direction as simag -> sibry
        on_axis = self._psi.ev(d["zmaxis"], d["rmaxis"]).item()
        if d["rbbbs"].size:
            outward = np.mean(self._psi.ev(d["zbbbs"], d["rbbbs"])) - on_axis
            self.flipped = bool(np.sign(outward) != np.sign(d["sibry"] - d["simag"]))
        else:
            self.flipped = abs(on_axis + d["simag"]) < abs(on_axis - d["simag"])
        if self.flipped:  # some writers store psirz with the opposite sign to simag and sibry
            self._psi = RectBivariateSpline(self.Z, self.R, -d["psirz"])
        # PCHIP: no overshoot where q grows very fast towards a separatrix
        psin = np.linspace(0.0, 1.0, d["nw"])
        self._q = PchipInterpolator(psin, np.abs(d["qpsi"]))
        self._int_q = self._q.antiderivative()
        self._p = PchipInterpolator(psin, d["pres"])
        self._F = PchipInterpolator(psin, np.abs(d["fpol"]))
        # psi between axis and boundary, per radian
        self.dpsi = abs(d["sibry"] - d["simag"]) / (2 * np.pi if cocos >= 11 else 1.0)

    # ---- profiles on psi_n
    def q(self, psin):
        return self._q(psin)

    def pressure(self, psin):
        return self._p(psin)

    def pressure_from_pprime(self, psin):
        """p from the boundary value of pres plus the integral of pprime, the source term of the file's G-S solution.
        The sign of the integral is taken so that p rises towards the axis, independent of the file's conventions."""
        d = self.data
        x = np.linspace(0.0, 1.0, d["nw"])
        integral = PchipInterpolator(x, d["pprime"] * (d["sibry"] - d["simag"])).antiderivative()
        drop = integral(1.0) - integral(np.asarray(psin, dtype=float))  # p(1) - p(psin) by the file's sign conventions
        sign = 1.0 if integral(1.0) - integral(0.0) < 0 else -1.0
        return d["pres"][-1] - sign * drop

    def F(self, psin):
        return self._F(psin)

    def psin(self, R, Z):
        d = self.data
        R, Z = np.broadcast_arrays(np.asarray(R, dtype=float), np.asarray(Z, dtype=float))
        psi = self._psi.ev(Z.ravel(), R.ravel()).reshape(R.shape)
        return (psi - d["simag"]) / (d["sibry"] - d["simag"])

    # ---- flux-coordinate maps
    def psit(self, psin):
        """Toroidal flux / 2 pi inside psi_n [Wb/rad]: integral of |q| dpsi_p."""
        return self.dpsi * self._int_q(psin)

    def rho(self, psin, psin_b):
        """DESC rho for a DESC boundary at psi_n = psin_b."""
        return np.sqrt(np.clip(self._int_q(psin) / self._int_q(psin_b), 0.0, None))

    def psin_of_rho(self, rho, psin_b):
        x = np.linspace(0.0, psin_b, 4001)
        return PchipInterpolator(self.rho(x, psin_b), x)(rho)

    def to_psin(self, coord, values):
        """psi_n of profile coordinates given as psi_n, rho_pol or rho_tor (see the module docstring)."""
        values = np.asarray(values, dtype=float)
        if coord == "psi_n":
            return values
        if coord == "rho_pol":
            return values**2
        if coord == "rho_tor":
            return self.psin_of_rho(values, 1.0)
        raise ValueError(f"unknown coordinate {coord!r}; use psi_n, rho_pol, rho_tor or rho")

    # ---- surfaces
    def trace(self, psin, n=1024):
        """R, Z of the surface psi_n = psin at n geometric angles around the magnetic axis (counter-clockwise)."""
        R0, Z0 = self.axis
        angle = 2 * np.pi * np.arange(n) / n
        c, s = np.cos(angle), np.sin(angle)
        with np.errstate(divide="ignore"):
            to_r = np.where(c > 0, (self.R[-1] - R0) / c, np.where(c < 0, (self.R[0] - R0) / c, np.inf))
            to_z = np.where(s > 0, (self.Z[-1] - Z0) / s, np.where(s < 0, (self.Z[0] - Z0) / s, np.inf))
        smax = 0.999 * np.minimum(to_r, to_z)
        frac = np.linspace(0.0, 1.0, 2001)[1:]
        dist = smax[:, None] * frac[None, :]
        values = self.psin(R0 + c[:, None] * dist, Z0 + s[:, None] * dist)
        R, Z = np.empty(n), np.empty(n)
        for k in range(n):
            hit = np.nonzero(values[k] >= psin)[0]
            if hit.size == 0 or hit[0] == 0:
                raise ValueError(f"psi_n = {psin} is not a closed surface around the axis "
                                 f"(ray at {np.degrees(angle[k]):.1f} deg); use a smaller boundary psi_n")
            j = hit[0]
            f = lambda t: float(self.psin(R0 + c[k] * t, Z0 + s[k] * t)) - psin
            t = brentq(f, dist[k, j - 1], dist[k, j], xtol=1e-12)
            R[k], Z[k] = R0 + c[k] * t, Z0 + s[k] * t
        return R, Z


def arc_length_theta(R, Z):
    """Poloidal angle proportional to arc length along a closed curve, starting at the first point."""
    seg = np.hypot(np.diff(np.append(R, R[0])), np.diff(np.append(Z, Z[0])))
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return 2 * np.pi * s / seg.sum()
