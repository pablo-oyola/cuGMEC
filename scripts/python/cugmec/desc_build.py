"""
gEQDSK -> fixed-boundary DESC equilibrium, as scripts/equilibrium/compute2D_0170.ipynb does for analytic inputs.

The DESC boundary is the g-file surface psi_n = psin_b. Use psin_b < 1 for diverted plasmas, since a Fourier boundary
cannot represent an X-point. iota = 1/|q| and p are fitted as even power series in DESC's rho, and Psi is the
toroidal flux inside that surface. After the solve, compare() checks the result against the g-file.
"""
import numpy as np

import desc.grid
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile

from .geqdsk import arc_length_theta


def _fit(rho, values, order):
    profile = PowerSeriesProfile.from_values(rho, values, order=order, sym=True)
    index = np.zeros((rho.size, 3))
    index[:, 0] = rho
    fitted = np.asarray(profile.compute(desc.grid.Grid(index)))
    return profile, fitted


def _fit_auto(rho, values, order, error, tolerance=1e-3, orders=range(8, 33, 4)):
    """Fixed order, or (order='auto') the lowest even order whose error(fitted) is below tolerance (else the best)."""
    if order != "auto":
        profile, fitted = _fit(rho, values, int(order))
        return profile, fitted, int(order)
    best = None
    for n in orders:
        profile, fitted = _fit(rho, values, n)
        e = error(fitted)
        if best is None or e < best[0]:
            best = (e, profile, fitted, n)
        if e <= tolerance:
            break
    return best[1], best[2], best[3]


def force_error(eq):
    """Volume average of |F| over the volume average of |grad(B^2/2mu0)|."""
    grid = desc.grid.QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    f = eq.compute(["|F|", "|grad(|B|^2)|/2mu0", "sqrt(g)"], grid=grid)
    w = np.abs(np.asarray(f["sqrt(g)"])) * grid.weights
    return float(np.sum(np.asarray(f["|F|"]) * w) / np.sum(np.asarray(f["|grad(|B|^2)|/2mu0"]) * w))


def solve(eq, solver="auto", verbose=1, maxiter=300, tolerance=1e-3):
    """
    direct: eq.solve, as compute2D_0170.ipynb does. continuation: DESC's automatic continuation.
    auto: direct, then continuation from the same starting point if the force error is above tolerance.
    """
    start = eq.copy()
    if solver in ("direct", "auto"):
        eq.solve(objective="force", verbose=verbose, maxiter=maxiter, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if solver == "direct" or force_error(eq) <= tolerance:
            return eq, "direct"
    return solve_continuation_automatic(start, objective="force", verbose=verbose)[-1], "continuation"


def build_desc(g, psin_b=0.995, M=16, iota_order="auto", pressure_order="auto", pressure=None, solver="auto", verbose=1):
    """
    g: GEqdsk. pressure: optional callable p(rho) [Pa] used instead of the g-file pressure (e.g. kinetic).
    Returns (eq, report), with eq solved and report a dict of fit errors and inputs.
    """
    rho = np.linspace(0.0, 1.0, 201)
    psin = g.psin_of_rho(rho, psin_b)
    q = g.q(psin)
    p = g.pressure(psin) if pressure is None else pressure(rho)
    p_scale = np.abs(p).max() or 1.0
    iota, iota_fit, iota_order = _fit_auto(rho, 1.0 / q, iota_order, lambda f: np.abs(1.0 / f - q).max() / q.min())
    pres, pres_fit, pressure_order = _fit_auto(rho, p, pressure_order, lambda f: np.abs(f - p).max() / p_scale)

    R, Z = g.trace(psin_b, 1024)
    R, Z = np.roll(R[::-1], 1), np.roll(Z[::-1], 1)  # clockwise from the outboard midplane: DESC's Z ~ -sin(theta)
    theta = arc_length_theta(R, Z)
    coords = np.column_stack([R, np.zeros_like(R), Z])
    surface = FourierRZToroidalSurface.from_values(coords, theta, M=M, N=0, NFP=1, sym=False)
    x = np.asarray(surface.compute("x", grid=desc.grid.Grid(np.column_stack([np.ones_like(theta), theta, 0 * theta])))["x"])
    bdry_err = np.hypot(R - x[:, 0], Z - x[:, 2]).max()

    Psi = 2 * np.pi * float(g.psit(psin_b))
    eq = Equilibrium(Psi=Psi, NFP=1, L=2 * M, M=M, N=0, pressure=pres, iota=iota, surface=surface, sym=False,
                     spectral_indexing="fringe")
    eq, used = solve(eq, solver, verbose)

    report = {
        "geqdsk": g.path, "cocos": g.cocos, "psin_b": psin_b, "M": M, "iota_order": iota_order,
        "pressure_order": pressure_order, "pressure_source": "geqdsk" if pressure is None else "kinetic", "solver": used,
        "Psi": Psi, "q_fit_max_rel_error": float(np.abs(1.0 / iota_fit - q).max() / q.min()),
        "p_fit_max_rel_error": float(np.abs(pres_fit - p).max() / p_scale),
        "boundary_fit_max_error_m": float(bdry_err),
    }
    return eq, report


def compare(eq, g, psin_b, rho_check=(0.25, 0.5, 0.75)):
    """Checks of a solved DESC equilibrium against the g-file it was built from."""
    out = {}
    axis = desc.grid.Grid(np.array([[0.0, 0.0, 0.0]]))
    a = eq.compute(["|B|", "R", "Z"], grid=axis)
    out["B_axis_desc"] = float(a["|B|"][0])
    out["B_axis_geqdsk"] = float(g.F(0.0) / g.axis[0])
    out["axis_shift_m"] = float(np.hypot(a["R"][0] - g.axis[0], a["Z"][0] - g.axis[1]))

    rho = np.linspace(0.05, 1.0, 40)
    grid = desc.grid.LinearGrid(rho=rho, M=eq.M_grid, N=0, NFP=1, sym=False)
    q_desc = 1.0 / grid.compress(eq.compute("iota", grid=grid)["iota"])
    q_g = g.q(g.psin_of_rho(rho, psin_b))
    out["q_max_rel_error"] = float(np.abs(q_desc - q_g).max() / q_g.min())

    # where DESC's flux surfaces fall in the g-file's psi_n
    grid = desc.grid.LinearGrid(rho=np.array(rho_check), M=64, N=0, NFP=1, sym=False)
    x = eq.compute(["R", "Z"], grid=grid)
    psin = g.psin(np.asarray(x["R"]), np.asarray(x["Z"]))
    target = g.psin_of_rho(np.array(rho_check), psin_b)
    out["surface_max_psin_error"] = {
        f"{r:g}": float(np.abs(psin[np.isclose(grid.nodes[:, 0], r)] - target[i]).max()) for i, r in enumerate(rho_check)}

    out["force_error"] = force_error(eq)
    return out
