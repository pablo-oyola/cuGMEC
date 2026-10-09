"""
standard2D directly from a gEQDSK, without re-solving the equilibrium.

pyCOCOS builds straight-field-line (PEST) coordinates from the g-file's own flux map psi(R, Z) and F(psi). Everything
here derives from that map, so the coordinates, |B|, B^y and q are mutually consistent:

    q            surface integrals of the traced flux surfaces (pyCOCOS); the file's qpsi is only compared against
    rho          DESC's sqrt(psi_t / psi_t(psin_b)), with psi_t the integral of q dpsi_p
    R, Z         the PEST map sampled on a fine (rho, theta) grid, then a Fourier series in theta whose coefficients
                 are least-squares Chebyshev series in rho, which gives derivatives to second order
    metric       g_rr = R_r^2 + Z_r^2, g_rt = R_r R_t + Z_r Z_t, g_tt = R_t^2 + Z_t^2, g_pp = R^2, and the inverse
    sqrt(g)      R (R_t Z_r - R_r Z_t)
    |B|          B^2 = (F^2 + psi_p'^2 g^rr) / R^2
    J_parallel   mu0 J_par = SIGN (dF/dpsi_p B + mu0 F dp/dpsi_p / B)        (Grad-Shafranov current)

theta follows DESC's orientation (Z ~ -sin theta, sqrt(g) > 0), with theta = 0 on the outboard midplane ray from the
axis. A PEST angle is defined up to theta + c(rho); cuGMEC's equations do not depend on that choice. The keys and
meanings are those of desc_eval.standard2d, so build_2d() treats both paths alike.
"""
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

MU0 = 4e-7 * np.pi
SIGN = -1.0  # sign of J_parallel relative to DESC's for the same equilibrium; checked against DESC on ITPA


class _Field:
    """f(rho, theta) as a Fourier series in theta whose coefficients are least-squares Chebyshev series in rho.

    The fit (rather than interpolation) filters the small radial noise of the traced surfaces, which second
    derivatives would otherwise amplify."""

    def __init__(self, rho, values, degree=12):
        self.degree = degree
        nt = values.shape[1]
        c = np.fft.rfft(values, axis=1) / nt
        m = np.arange(c.shape[1])
        keep = m < nt / 2  # no Nyquist term
        c = (c * np.where(m == 0, 1.0, 2.0))[:, keep]
        self.m = m[keep]
        self.lo, self.hi = float(rho[0]), float(rho[-1])
        V = np.polynomial.chebyshev.chebvander(self._x(rho), degree)
        data = np.concatenate([c.real, c.imag], axis=1)
        self.coef = np.linalg.lstsq(V, data, rcond=None)[0]
        self.residual = float(np.abs(V @ self.coef - data).max())

    def _x(self, rho):
        return (2 * np.asarray(rho, dtype=float) - self.lo - self.hi) / (self.hi - self.lo)

    def __call__(self, rho, theta, dr=0, dt=0):
        coef = self.coef
        if dr:
            coef = np.polynomial.chebyshev.chebder(coef, m=dr, axis=0) * (2 / (self.hi - self.lo)) ** dr
        c = np.polynomial.chebyshev.chebvander(self._x(rho), coef.shape[0] - 1) @ coef
        n = self.m.size
        c = (c[:, :n] + 1j * c[:, n:]) * (1j * self.m) ** dt
        return np.real(c @ np.exp(1j * np.outer(self.m, theta)))


class _Cheb1D:
    """Least-squares Chebyshev series of a radial profile, with derivatives."""

    def __init__(self, x, y, degree, lo, hi):
        self.degree = degree
        self.series = np.polynomial.Chebyshev.fit(x, y, degree, domain=[lo, hi])
        self.residual = float(np.abs(self.series(x) - y).max())

    def __call__(self, x, dr=0):
        return (self.series.deriv(dr) if dr else self.series)(np.asarray(x, dtype=float))


def _plateau(build, degrees):
    """The lowest degree whose fit residual is within 1.5x of the best: past it, a higher degree fits noise."""
    fits = {d: build(d) for d in degrees}
    best = min(f.residual for f in fits.values())
    return next(fits[d] for d in degrees if fits[d].residual <= 1.5 * best)


def _repaired(g):
    """Path of the g-file for pyCOCOS: a copy with psirz negated when it contradicts simag/sibry (GEqdsk.flipped)."""
    if not g.flipped:
        return g.path
    from freeqdsk import geqdsk

    with open(g.path) as f:
        d = geqdsk.read(f)
    d["psi"] = -np.asarray(d["psi"])
    out = Path(tempfile.mkdtemp(prefix="cugmec_")) / (Path(g.path).name + ".repaired")
    with open(out, "w") as f:
        geqdsk.write(d, f)
    return str(out)


class PestMap:
    """PEST coordinates and the flux map of a g-file, from pyCOCOS, out to psi_n = psin_b (DESC's rho = 1)."""

    def __init__(self, g, psin_b=0.995, lpsi=128, n_theta=256, spectral_max_mode=32):
        from pycocos import EQDSK

        self.g, self.psin_b, self.n_theta = g, psin_b, n_theta
        eq = EQDSK(_repaired(g), cocos_in=g.cocos, cocos_internal=1)
        self.psi_ax = float(eq.geometry.attrs.get("psi_ax", eq._psi_ax_init))
        self.psi_b = float(eq.geometry.attrs.get("psi_bdy", eq._psi_edge_init))
        # up to the LCFS for limited plasmas (psin_b = 1); short of the separatrix otherwise
        rhopol_max = 0.9999 if psin_b >= 1.0 else min(0.999, np.sqrt(psin_b) + 0.02)
        self.product = eq.compute_coordinates("pest", lpsi=lpsi, ltheta=n_theta, rhopol_min=0.02, rhopol_max=rhopol_max,
                                              spectral_max_mode=spectral_max_mode, materialize_rz=False)
        psin = (self.product.psi - self.psi_ax) / (self.psi_b - self.psi_ax)
        order = np.argsort(psin)  # pyCOCOS sorts by psi, which may decrease outward
        psin, q = psin[order], np.abs(self.product.q[order])
        inside = (psin > 0) & (psin <= 1)
        self.psin_s, self.q_s = psin[inside], q[inside]
        self.q_extrapolated_from = float(self.psin_s[-1]) if self.psin_s[-1] < psin_b else None
        if self.psin_s[-1] < psin_b - 0.005:
            raise ValueError(f"pyCOCOS surfaces end at psi_n = {self.psin_s[-1]:.4f}, short of psin_b = {psin_b}")
        q = PchipInterpolator(self.psin_s, self.q_s, extrapolate=True)  # extrapolated only below psi_n ~ 4e-4
        self._int_q = q.antiderivative()
        self.psit_b = float(g.dpsi * (self._int_q(psin_b) - self._int_q(0.0)))

    # ---- flux map consistent with the traced surfaces
    def rho(self, psin):
        """DESC rho of psi_n."""
        z = self._int_q(0.0)
        x = np.asarray(psin, dtype=float)
        return np.sqrt(np.clip((self._int_q(x) - z) / (self._int_q(self.psin_b) - z), 0.0, None))

    def psin_of_rho(self, rho):
        x = np.linspace(0.0, self.psin_b, 4001)
        return PchipInterpolator(self.rho(x), x)(np.asarray(rho, dtype=float))

    def standard2d(self, rho_min, rho_max, rho_num, theta_num, pressure=None, n_surfaces=128, radial_degree="auto",
                   iota_degree="auto"):
        """
        The contents of standard2D.mat on rho_num uniform rho in [rho_min, rho_max] x theta_num cell-centred theta.
        pressure: callable p(rho) [Pa], or None for the g-file pres.
        radial_degree, iota_degree: Chebyshev degrees in rho; auto takes the start of the residual plateau.
        Returns (S, report).
        """
        g = self.g
        lo, hi = max(0.02, rho_min - 0.05), min(1.0, rho_max + 0.05)

        # iota(rho) fitted to the surface q values themselves, for smooth first and second derivatives
        rho_q = self.rho(self.psin_s)
        sel = (rho_q >= lo - 0.02) & (rho_q <= hi + 0.02)
        fit_iota = lambda d: _Cheb1D(rho_q[sel], 1.0 / self.q_s[sel], d, lo - 0.02, hi + 0.02)
        iota = fit_iota(int(iota_degree)) if iota_degree != "auto" else _plateau(fit_iota, range(8, min(41, sel.sum() - 4), 4))
        dpsip = lambda r: 2 * self.psit_b * r * iota(r)  # dpsi_p/drho (Wb/rad)

        rho1 = np.linspace(0.0, 1.0, 201)
        psin1 = self.psin_of_rho(rho1)
        F = CubicSpline(rho1, g.F(psin1))
        p = CubicSpline(rho1, g.pressure(psin1) if pressure is None else pressure(rho1))

        # PEST map on the fine grid
        rho_s = np.linspace(lo, hi, n_surfaces)
        theta_s = 2 * np.pi * np.arange(self.n_theta) / self.n_theta
        psi_s = self.psi_ax + self.psin_of_rho(rho_s) * (self.psi_b - self.psi_ax)
        for flip in (1.0, -1.0):  # theta must run as in DESC (sqrt(g) > 0)
            v = self.product.values(psi_s[:, None], flip * theta_s[None, :])
            if radial_degree == "auto":
                R = _plateau(lambda d: _Field(rho_s, v["R"], d), range(12, 33, 4))
                Z = _plateau(lambda d: _Field(rho_s, v["z"], d), range(12, 33, 4))
                degree = max(R.degree, Z.degree)
            else:
                degree = int(radial_degree)
            R, Z = _Field(rho_s, v["R"], degree), _Field(rho_s, v["z"], degree)
            D = R(rho_s, theta_s, 0, 1) * Z(rho_s, theta_s, 1, 0) - R(rho_s, theta_s, 1, 0) * Z(rho_s, theta_s, 0, 1)
            if np.median(D) > 0:
                break

        Rs, Rr, Rt = R(rho_s, theta_s), R(rho_s, theta_s, 1, 0), R(rho_s, theta_s, 0, 1)
        Zr, Zt = Z(rho_s, theta_s, 1, 0), Z(rho_s, theta_s, 0, 1)
        D = Rt * Zr - Rr * Zt
        pp, Fs = dpsip(rho_s)[:, None], F(rho_s)[:, None]
        Bs = np.sqrt(Fs**2 + pp**2 * (Rt**2 + Zt**2) / D**2) / Rs
        dF, dp = F(rho_s, 1)[:, None] / pp, p(rho_s, 1)[:, None] / pp
        B = _Field(rho_s, Bs, degree)
        Jpar = _Field(rho_s, SIGN * (dF * Bs + MU0 * Fs * dp / Bs) / MU0, degree)

        # cuGMEC grid (as getRhoThetaPEST in the notebook)
        rho = np.linspace(rho_min, rho_max, rho_num)
        theta = -np.pi + (np.arange(theta_num) + 0.5) * 2 * np.pi / theta_num
        d = {f"{name}{a}": f(rho, theta, *nu) for name, f in (("R", R), ("Z", Z))
             for a, nu in (("", (0, 0)), ("r", (1, 0)), ("t", (0, 1)), ("rr", (2, 0)), ("rt", (1, 1)), ("tt", (0, 2)))}
        R0, Rr, Rt, Rrr, Rrt, Rtt = (d[k] for k in ("R", "Rr", "Rt", "Rrr", "Rrt", "Rtt"))
        Z0, Zr, Zt, Zrr, Zrt, Ztt = (d[k] for k in ("Z", "Zr", "Zt", "Zrr", "Zrt", "Ztt"))
        D = Rt * Zr - Rr * Zt
        D_r = Rrt * Zr + Rt * Zrr - Rrr * Zt - Rr * Zrt
        D_t = Rtt * Zr + Rt * Zrt - Rrt * Zt - Rr * Ztt

        zero = np.zeros((rho_num, theta_num))
        cov = {"xx": Rr**2 + Zr**2, "xy": Rr * Rt + Zr * Zt, "yy": Rt**2 + Zt**2, "zz": R0**2, "xz": zero, "yz": zero}
        dcov = {
            "px": {"xx": 2 * (Rr * Rrr + Zr * Zrr), "xy": Rrr * Rt + Rr * Rrt + Zrr * Zt + Zr * Zrt,
                   "yy": 2 * (Rt * Rrt + Zt * Zrt), "zz": 2 * R0 * Rr, "xz": zero, "yz": zero},
            "py": {"xx": 2 * (Rr * Rrt + Zr * Zrt), "xy": Rrt * Rt + Rr * Rtt + Zrt * Zt + Zr * Ztt,
                   "yy": 2 * (Rt * Rtt + Zt * Ztt), "zz": 2 * R0 * Rt, "xz": zero, "yz": zero},
        }
        con = {"xx": cov["yy"] / D**2, "xy": -cov["xy"] / D**2, "yy": cov["xx"] / D**2, "zz": 1 / R0**2,
               "xz": zero, "yz": zero}
        dcon = {}
        for ax, Dx, Rx in (("px", D_r, Rr), ("py", D_t, Rt)):
            quot = lambda num, dnum: dnum / D**2 - 2 * num * Dx / D**3
            dcon[ax] = {"xx": quot(cov["yy"], dcov[ax]["yy"]), "xy": -quot(cov["xy"], dcov[ax]["xy"]),
                        "yy": quot(cov["xx"], dcov[ax]["xx"]), "zz": -2 * Rx / R0**3, "xz": zero, "yz": zero}

        rep = lambda f: np.repeat(np.asarray(f)[:, None], theta_num, axis=1)
        S = {"rho": rep(rho), "theta_pest": np.repeat(theta[None, :], rho_num, axis=0)}
        S["theta_desc"] = S["theta_pest"]
        S.update(R=R0, Z=Z0, dR_drho=Rr, dR_dtheta=Rt, dZ_drho=Zr, dZ_dtheta=Zt)
        S.update(JPEST=R0 * D, dJPEST_drho=Rr * D + R0 * D_r, dJPEST_dtheta=Rt * D + R0 * D_t,
                 J_parallel=Jpar(rho, theta))
        S.update(B=B(rho, theta), dB_dr=B(rho, theta, 1, 0), dB_dt=B(rho, theta, 0, 1), dB_dr2=B(rho, theta, 2, 0),
                 dB_drdt=B(rho, theta, 1, 1), dB_dt2=B(rho, theta, 0, 2))
        for k in ("lambda", "dlambda_dr", "dlambda_dt", "dlambda_dr2", "dlambda_drdt", "dlambda_dt2"):
            S[k] = zero  # theta is already the PEST angle
        for comp in cov:
            S[f"PESTcov_{comp}"], S[f"PESTcon_{comp}"] = cov[comp], con[comp]
            for ax in ("px", "py"):
                S[f"dPESTcov_{comp}_{ax}"], S[f"dPESTcon_{comp}_{ax}"] = dcov[ax][comp], dcon[ax][comp]
        S.update(iota=rep(iota(rho)), iota_r=rep(iota(rho, 1)), iota_rr=rep(iota(rho, 2)), q=rep(1 / iota(rho)))
        S.update(diota_dr=S["iota_r"], diota_dr2=S["iota_rr"])
        S.update(psip=rep(self.psin_of_rho(rho) * g.dpsi), dpsip_dr=rep(dpsip(rho)))
        S.update(B00=np.array([g.F(0.0) / g.axis[0]]), L00=np.array([g.axis[0]]), psit_max=np.array([self.psit_b]))

        # checks: straight field lines, the fits, and the file's own qpsi against its flux map
        sfl = (1 / iota(rho))[:, None] * dpsip(rho)[:, None] * R0**2 / F(rho)[:, None]
        inside = self.psin_s < self.psin_b
        q_file = g.q(self.psin_s[inside])
        report = {
            "method": "pycocos", "psin_b": self.psin_b, "psit_b": self.psit_b,
            "pest_jacobian_max_rel_error": float(np.max(np.abs(S["JPEST"] - sfl) / np.abs(S["JPEST"]))),
            "radial_degree": degree, "radial_fit_residual_m": max(R.residual, Z.residual),
            "iota_degree": iota.degree, "iota_fit_max_rel_error": float(iota.residual * self.q_s.max()),
            "geqdsk_qpsi_vs_flux_map_max_rel_error": float(np.max(np.abs(q_file - self.q_s[inside]) / self.q_s[inside])),
            "theta_reversed": bool(flip < 0), "psirz_sign_repaired": bool(g.flipped),
            "q_extrapolated_from_psin": self.q_extrapolated_from,
        }
        return S, report
