"""
Equilibrium blocks read by cuGMEC, and the Python replacement for the MATLAB stage of the 2D (tokamak) preprocessing.

    standard2D.mat (scripts/equilibrium/output2D_0170.ipynb) + NTP.mat (profiles)
        -> 159 field blocks  (what writeProfile1D, DESC2PEST2SFA2D and writeEquilibrium2D produce)
        -> normalization     (what writeNormalization2D produces)

The binary cuGMEC reads is a headerless little-endian float64 stream. Its block order and shapes mirror
loadMHDEquilibrium() in src/cuGMEC_model.h. Every block is stored in C order (x, y, z).
"""
import numpy as np
from scipy.interpolate import make_interp_spline

MU0 = 4e-7 * np.pi
MP = 1.672621637e-27
QE = 1.6021766208e-19
KEV = 1e3 * QE
GHOST = 2

MAXWELL, SLOWING, ABSENT = 1, 2, 3  # species type codes used by generateInput2D.m

MET = ["xx", "xy", "yy", "xz", "yz", "zz"]  # component order of metric_seq_ in writeEquilibrium2D.m
IJ = {"xx": (0, 0), "xy": (0, 1), "yy": (1, 1), "xz": (0, 2), "yz": (1, 2), "zz": (2, 2)}
D4 = ["", "_px", "_py", "_pz"]
D10 = ["", "_px", "_py", "_pz", "_px2", "_pxy", "_pxz", "_py2", "_pyz", "_pz2"]


# ---------------------------------------------------------------- file layout

def layout():
    """(name, ghosted) for each of the 159 blocks, in file order. Ghosted blocks have gridNy + 4 points in y."""
    blocks = [("qtheta", True)]
    blocks += [(f"SFA{cc}{m}{d}", True) for cc in ("con", "cov") for m in MET for d in D4]
    for f in ["q", "psip", "Ni", "Ti", "Pi", "Ne", "Te", "Pe", "Na", "Ta", "Nb", "Tb"]:
        blocks += [(f, False), (f + "_px", False)]
    blocks += [(f"g{cc}{m}{d}", False) for cc in ("con", "cov") for m in MET for d in D4]
    for f in ["J", "Bny", "JpB", "Rho", "Va", "R", "Z"]:
        blocks += [(f + d, False) for d in D4]
    blocks += [("B" + d, False) for d in D10]
    return blocks


def expected_doubles(Nx, Ny, equilNz=1):
    return Nx * equilNz * (49 * (Ny + 2 * GHOST) + 110 * Ny)


def read_bin(path, Nx, Ny, equilNz=1):
    """Read an equilibrium file into {block name: (Nx, ny, equilNz) array}. Refuses a size mismatch."""
    data = np.fromfile(path, dtype="<f8")
    n = expected_doubles(Nx, Ny, equilNz)
    if data.size != n:
        raise ValueError(f"{path}: {data.size} doubles, expected {n} for Nx={Nx} Ny={Ny} equilNz={equilNz}")
    out, off = {}, 0
    for name, ghosted in layout():
        ny = Ny + 2 * GHOST if ghosted else Ny
        size = Nx * ny * equilNz
        out[name] = data[off:off + size].reshape(Nx, ny, equilNz)
        off += size
    return out


def write_bin(path, blocks, Nx, Ny, equilNz=1):
    """Write blocks given as (Nx, ny) or (Nx, ny, equilNz) arrays; 2D arrays are repeated along z."""
    parts = []
    for name, ghosted in layout():
        ny = Ny + 2 * GHOST if ghosted else Ny
        a = np.asarray(blocks[name], dtype="<f8")
        if a.shape == (Nx, ny):
            a = a[:, :, None]
        parts.append(np.broadcast_to(a, (Nx, ny, equilNz)).ravel(order="C"))
    np.concatenate(parts).tofile(path)


# ---------------------------------------------------------------- profiles (port of writeProfile1D.m)

def sliding_poly(x, y, x2, window, order):
    """Fit a polynomial of the given order on every window of points (step 1). At each target point, use
    the covering window whose centre is nearest. Returns the values and d/dx at x2."""
    x, y = np.ravel(x), np.ravel(y)
    polys, lo, hi = [], [], []
    for i in range(x.size - window + 1):
        xs, ys = x[i:i + window], y[i:i + window]
        polys.append(np.polyfit(xs, ys, order))
        lo.append(xs.min())
        hi.append(xs.max())
    lo, hi = np.asarray(lo), np.asarray(hi)
    centre = 0.5 * (lo + hi)
    f, df = np.empty_like(x2, dtype=float), np.empty_like(x2, dtype=float)
    for k, xk in enumerate(x2):
        ok = np.where((xk >= lo) & (xk <= hi))[0]
        if ok.size == 0:
            raise ValueError(f"rho = {xk} lies outside the profile samples")
        p = polys[ok[np.argmin(np.abs(centre[ok] - xk))]]
        f[k] = np.polyval(p, xk)
        df[k] = np.polyval(np.polyder(p), xk) if order > 0 else 0.0
    return f, df


# ---------------------------------------------------------------- coordinates (replaces the SFT/SFA blocks of DESC2PEST2SFA2D.m)

def field_aligned_metric(gcon, dgcon, gcov, dgcov, q, dq, d2q, s, Drho):
    """
    Transform the PEST metric (rho, theta, phi) to x = (rho - rho0)/Drho, y = theta, z = phi - q(rho)*(theta - theta_ref).

    s = theta - theta_ref is theta_PEST for the field-aligned set (SFA) and 0 for the shifted metric (SFT).
    gcon, gcov have shape (..., 3, 3). dgcon, dgcov map 'r', 't' to their d/drho and d/dtheta; d/dphi is zero (axisymmetric).
    Returns g^ij, g_ij and their derivatives along x, y, z. s is differentiated before it is set (ds/dtheta = 1).
    """
    Z, O = np.zeros(q.shape), np.ones(q.shape)
    mat = lambda rows: np.stack([np.stack(r, -1) for r in rows], -2)
    A = mat([[O / Drho, Z, Z], [Z, O, Z], [-dq * s, -q, O]])          # d(x,y,z)/d(rho,theta,phi)
    dA_r = mat([[Z, Z, Z], [Z, Z, Z], [-d2q * s, -dq, Z]])
    dA_t = mat([[Z, Z, Z], [Z, Z, Z], [-dq, Z, Z]])
    Bm = mat([[Drho * O, Z, Z], [Z, O, Z], [Drho * dq * s, q, O]])    # inverse of A
    dB_r = mat([[Z, Z, Z], [Z, Z, Z], [Drho * d2q * s, dq, Z]])
    dB_t = mat([[Z, Z, Z], [Z, Z, Z], [Drho * dq, Z, Z]])

    T = lambda M: np.swapaxes(M, -1, -2)
    con = A @ gcon @ T(A)
    cov = T(Bm) @ gcov @ Bm
    dcon = {c: dA @ gcon @ T(A) + A @ dgcon[c] @ T(A) + A @ gcon @ T(dA) for c, dA in (("r", dA_r), ("t", dA_t))}
    dcov = {c: T(dB) @ gcov @ Bm + T(Bm) @ dgcov[c] @ Bm + T(Bm) @ gcov @ dB for c, dB in (("r", dB_r), ("t", dB_t))}
    # d/dx = Drho*(d/drho + q' s d/dphi), d/dy = d/dtheta + q d/dphi, d/dz = d/dphi, with d/dphi = 0
    zero = np.zeros_like(con)
    return con, cov, {"x": Drho * dcon["r"], "y": dcon["t"], "z": zero}, {"x": Drho * dcov["r"], "y": dcov["t"], "z": zero}


def sym3(d, prefix, suffix=""):
    """Assemble a (..., 3, 3) symmetric metric from keys such as PESTcon_xx, PESTcon_xy, ..."""
    g = np.empty(d[f"{prefix}_xx{suffix}"].shape + (3, 3))
    for m, (i, j) in IJ.items():
        g[..., i, j] = g[..., j, i] = d[f"{prefix}_{m}{suffix}"]
    return g


# ---------------------------------------------------------------- equilibrium file

def build_2d(S, P, ion_mass, ion_type=MAXWELL, alpha_type=SLOWING, beam_type=SLOWING, fit=None):
    """
    S: contents of standard2D.mat. P: contents of NTP.mat. ion_mass: thermal ion mass / m_p.
    *_type: MAXWELL, SLOWING or ABSENT, matching if<Species> and <Species>Type in cuGMEC_param.h.
    fit: {"ne": (window, order), ...} for the profile fits. The default is order 12 over all samples.
    Returns (blocks, norm), where norm holds the values for cuGMEC_param.h.
    """
    if ion_type == ABSENT:
        raise ValueError("thermal ions are required: VA0, Rho and Va are built from n_i and p_i")
    fit = fit or {}
    rho = S["rho"]
    Nx, Ny = rho.shape
    rho1 = rho[:, 0]
    Drho = rho1[-1] - rho1[0]
    q, iota = S["q"], S["iota"]
    dq = -S["diota_dr"] / iota**2
    d2q = -S["diota_dr2"] / iota**2 + 2 * S["diota_dr"] ** 2 / iota**3

    B0, L0, psit = (float(np.ravel(S[k])[0]) for k in ("B00", "L00", "psit_max"))
    VA0 = B0 / np.sqrt(MU0 * ion_mass * MP * np.ravel(P["niSample"])[0] * 1e19)
    P0 = B0**2 / (2 * MU0)
    T0, N0 = P0 * L0**3, L0**-3
    mi = ion_mass * MP

    rho_s = np.ravel(P["rhoSample"])

    def profile(name):
        window, order = fit.get(name, (rho_s.size, 12))
        return sliding_poly(rho_s, P[name + "Sample"], rho1, window, order)

    out = {}
    rep = lambda v: np.repeat(v[:, None], Ny, 1)
    Z0 = np.zeros((Nx, Ny))

    def radial(name, f, df, f0):
        out[name], out[name + "_px"] = rep(f / f0), rep(Drho * df / f0)

    def full(name, f, fr, ft, f0):
        out[name], out[name + "_px"], out[name + "_py"], out[name + "_pz"] = f / f0, fr * Drho / f0, ft / f0, Z0

    # ---- 1D profiles
    radial("q", q[:, 0], dq[:, 0], 1.0)
    radial("psip", S["psip"][:, 0], S["dpsip_dr"][:, 0], B0 * L0**2)
    ne, dne = (1e19 * v for v in profile("ne"))
    Te, dTe = (KEV * v for v in profile("Te"))
    radial("Ne", ne, dne, N0)
    radial("Te", Te, dTe, T0)
    radial("Pe", ne * Te, dne * Te + dTe * ne, P0)

    pressure = {}
    for tag, kind, N, T in (("i", ion_type, "Ni", "Ti"), ("a", alpha_type, "Na", "Ta"), ("b", beam_type, "Nb", "Tb")):
        if kind == ABSENT:
            z = np.zeros(Nx)
            radial(N, z, z, N0)
            radial(T, z, z, 1.0)
            continue
        n, dn = (1e19 * v for v in profile("n" + tag))
        t, dt = profile("T" + tag)
        radial(N, n, dn, N0)
        if kind == MAXWELL:
            radial(T, t, dt, 1.0)                           # keV, not normalized
            pressure[tag] = (n * t * KEV, (dn * t + dt * n) * KEV, n, dn)
        else:
            radial(T, t, dt, VA0)                           # critical speed v_c in m/s
            pressure[tag] = (*profile("P" + tag), n, dn)
    radial("Pi", pressure["i"][0], pressure["i"][1], P0)  # alpha and beam pressures come from PIC, not the file

    # ---- |B| in PEST coordinates: chain rule from theta_DESC to theta_PEST = theta + lambda
    lr, lt, lrr, lrt, ltt = (S[k] for k in ("dlambda_dr", "dlambda_dt", "dlambda_dr2", "dlambda_drdt", "dlambda_dt2"))
    B, Br, Bt, Brr, Brt, Btt = (S[k] for k in ("B", "dB_dr", "dB_dt", "dB_dr2", "dB_drdt", "dB_dt2"))
    D = 1 + lt
    B_r = Br - Bt * lr / D
    B_t = Bt / D
    B_rr = D**-3 * (Btt * lr**2 * D + D**2 * (-2 * Brt * lr + Brr * D) - Bt * (D * (-2 * lr * lrt + lrr * D) + lr**2 * ltt))
    B_rt = D**-3 * (-Btt * lr * D + Brt * D**2 + Bt * (-lrt * D + lr * ltt))
    B_tt = D**-3 * (Btt * D - Bt * ltt)
    for d, v in zip(D10, [B, B_r * Drho, B_t, Z0, B_rr * Drho**2, B_rt * Drho, Z0, B_tt, Z0, Z0]):
        out["B" + d] = v / B0

    # ---- Jacobian of (x, y, z) and contravariant B^y
    J, Jr, Jt = (Drho * S[k] for k in ("JPEST", "dJPEST_drho", "dJPEST_dtheta"))
    full("J", J, Jr, Jt, L0**3)
    Bny = 2 * psit * Drho * rho / q / J
    Bny_r = 2 * psit * Drho * (1 / q / J - rho * dq / J / q**2 - rho * Jr / q / J**2)
    Bny_t = 2 * psit * Drho * (-rho * Jt / q / J**2)
    full("Bny", Bny, Bny_r, Bny_t, B0 / L0)

    # ---- J_parallel / B. MATLAB uses a 5th-order B-spline; here a spline in rho and an FFT in theta (periodic, uniform)
    jp = S["J_parallel"]
    jp_r = make_interp_spline(rho1, jp, k=5, axis=0).derivative()(rho1)
    m = np.fft.fftfreq(Ny, 1.0 / Ny)
    jp_t = np.real(np.fft.ifft(1j * m * np.fft.fft(jp, axis=1), axis=1))
    full("JpB", jp / B, (jp_r * B - B_r * jp) / B**2, (jp_t * B - B_t * jp) / B**2, 1 / (MU0 * L0))

    # ---- thermal-ion Larmor radius and local Alfven speed
    p_i, dp_i, n_i, dn_i = (rep(v) for v in pressure["i"])
    Rho = np.sqrt(mi * p_i / n_i) / QE / B
    Rho_r = np.sqrt(mi) / QE * (0.5 * (p_i / n_i) ** -0.5 * (dp_i / n_i - dn_i * p_i / n_i**2) * B - B_r * np.sqrt(p_i / n_i)) / B**2
    Rho_t = -np.sqrt(mi * p_i / n_i) / QE / B**2 * B_t
    full("Rho", Rho, Rho_r, Rho_t, L0)
    a = MU0 * mi * n_i
    full("Va", B / np.sqrt(a), B_r / np.sqrt(a) - 0.5 * B * a**-1.5 * MU0 * mi * dn_i, B_t / np.sqrt(a), VA0)
    full("R", S["R"], S["dR_drho"], S["dR_dtheta"], L0)
    full("Z", S["Z"], S["dZ_drho"], S["dZ_dtheta"], L0)

    # ---- metrics: shifted (SFT, s = 0) and field-aligned on the ghost-extended theta grid (SFA, s = theta)
    gcon, gcov = sym3(S, "PESTcon"), sym3(S, "PESTcov")
    dgcon = {"r": sym3(S, "dPESTcon", "_px"), "t": sym3(S, "dPESTcon", "_py")}  # '_px' in standard2D.mat is d/drho
    dgcov = {"r": sym3(S, "dPESTcov", "_px"), "t": sym3(S, "dPESTcov", "_py")}

    def emit(prefix, con, cov, dcon, dcov):
        for cc, g, dg, g0 in (("con", con, dcon, L0**-2), ("cov", cov, dcov, L0**2)):
            for comp, (i, j) in IJ.items():
                out[f"{prefix}{cc}{comp}"] = g[..., i, j] / g0
                for ax in "xyz":
                    out[f"{prefix}{cc}{comp}_p{ax}"] = dg[ax][..., i, j] / g0

    emit("g", *field_aligned_metric(gcon, dgcon, gcov, dgcov, q, dq, d2q, np.zeros_like(q), Drho))

    Nyg = Ny + 2 * GHOST
    pad = lambda M: np.concatenate([M[:, -GHOST:], M, M[:, :GHOST]], axis=1)
    theta = np.broadcast_to(-np.pi + (np.arange(Nyg) - (GHOST - 0.5)) / Ny * 2 * np.pi, (Nx, Nyg))
    qg, dqg, d2qg = (np.repeat(v[:, :1], Nyg, 1) for v in (q, dq, d2q))
    emit("SFA", *field_aligned_metric(pad(gcon), {c: pad(v) for c, v in dgcon.items()}, pad(gcov),
                                      {c: pad(v) for c, v in dgcov.items()}, qg, dqg, d2qg, theta, Drho))
    out["qtheta"] = qg * theta

    norm = dict(B0=B0, L0=L0, VA0=VA0, RHO0=rho1[0], RHO1=rho1[-1], PSITMAX=psit, gridNx=Nx, gridNy=Ny, NFP=1)
    for tag, name in (("i", "Ion"), ("a", "Alpha"), ("b", "Beam")):
        norm[name + "Beta"] = pressure[tag][0][0] / P0 if tag in pressure else 0.0  # at rho = RHO0, as in MATLAB
    return out, norm


# ---------------------------------------------------------------- block descriptions (written into HDF5 attributes)

_DERIV = {"": "", "_px": "d/dx of ", "_py": "d/dy of ", "_pz": "d/dz of ", "_px2": "d2/dx2 of ", "_pxy": "d2/dxdy of ",
          "_pxz": "d2/dxdz of ", "_py2": "d2/dy2 of ", "_pyz": "d2/dydz of ", "_pz2": "d2/dz2 of "}
_FIELDS = {
    "qtheta": ("q(rho) * theta_PEST on the ghost-extended y grid", "1"),
    "q": ("safety factor", "1"),
    "psip": ("poloidal flux / 2 pi [Wb/rad]", "B0 L0^2"),
    "Ni": ("thermal ion density [m^-3]", "L0^-3"),
    "Ne": ("electron density [m^-3]", "L0^-3"),
    "Na": ("alpha-slot density [m^-3]", "L0^-3"),
    "Nb": ("beam-slot density [m^-3]", "L0^-3"),
    "Ti": ("thermal ion temperature [keV] if Maxwellian, critical speed [m/s] if slowing-down", "1 (keV) or VA0"),
    "Ta": ("alpha-slot temperature [keV] if Maxwellian, critical speed [m/s] if slowing-down", "1 (keV) or VA0"),
    "Tb": ("beam-slot temperature [keV] if Maxwellian, critical speed [m/s] if slowing-down", "1 (keV) or VA0"),
    "Te": ("electron temperature [J]", "B0^2/(2 mu0) L0^3"),
    "Pi": ("thermal ion pressure [Pa]", "B0^2/(2 mu0)"),
    "Pe": ("electron pressure [Pa]", "B0^2/(2 mu0)"),
    "J": ("Jacobian of (x, y, z) [m^3]", "L0^3"),
    "Bny": ("contravariant B^y = 2 psi_t,edge Drho rho / (q J)", "B0/L0"),
    "JpB": ("J_parallel / |B|", "1/(mu0 L0)"),
    "Rho": ("thermal ion Larmor radius sqrt(m_i p_i / n_i) / (e B) [m]", "L0"),
    "Va": ("local Alfven speed B / sqrt(mu0 m_i n_i) [m/s]", "VA0"),
    "R": ("major radius [m]", "L0"),
    "Z": ("vertical position [m]", "L0"),
    "B": ("|B| [T]", "B0"),
}


def describe_block(name):
    """(description, normalization) of one block, e.g. 'SFAconxy_px' or 'Te'."""
    for suffix in sorted(_DERIV, key=len, reverse=True):
        if suffix and name.endswith(suffix):
            base, prefix = name[: -len(suffix)], _DERIV[suffix]
            break
    else:
        base, prefix = name, ""
    for tag, label, cov in (("SFAcon", "field-aligned (SFA) contravariant metric g^", False),
                            ("SFAcov", "field-aligned (SFA) covariant metric g_", True),
                            ("gcon", "shifted-metric (SFT) contravariant metric g^", False),
                            ("gcov", "shifted-metric (SFT) covariant metric g_", True)):
        if base.startswith(tag) and base[len(tag):] in IJ:
            return prefix + label + base[len(tag):], "L0^2" if cov else "L0^-2"
    text, norm = _FIELDS[base]
    return prefix + text, norm
