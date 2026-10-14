"""
Figures and numbers of the ITPA example (docs/figures/itpa/), from a cuGMEC run converted with `cugmec outputs`.

    python docs/scripts/make_figures.py [RUN_H5]        (default: TCV/ITPA_ref/ITPA_ref.h5, not tracked by git)

Writes the PNG figures and docs/figures/itpa/numbers.json (the values quoted in the text).
"""
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures" / "itpa"

# categorical slots 1-3 (validated together), text and grid inks, diverging blue - gray - red
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1d1d"])
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "font.size": 10, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


def itpa_fast_density(rho):
    """ITPA fast-ion density [m^-3] (Koenies et al. 2018) on rho = sqrt(s), s the normalized toroidal flux."""
    n0, c0, c1, c2, c3 = 1.44131e17, 0.49123, 0.298228, 0.198739, 0.521298
    return n0 * c3 * np.exp(-c2 / c1 * np.tanh((rho - c0) / c2))


def main(path):
    FIG.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "r")
    norm = f["equilibrium/normalization"]
    L0, VA0 = float(norm["L0"][()]), float(norm["VA0"][()])
    t0 = L0 / VA0
    t = f["time/t_diag"][()]
    rho = f["grid/rho"][()]
    A = f["diagnostics/amplitude"][:, :, 0]
    z = f["diagnostics/RealMode"][:, :, 0] + 1j * f["diagnostics/ImagMode"][:, :, 0]
    amax = A.max(1)
    late = t >= 0.6 * t[-1]
    g, b = np.polyfit(t[late], np.log(amax[late]), 1)
    k = int(A[-1].argmax())
    w = -np.polyfit(t[late], np.unwrap(np.angle(z[late, k])), 1)[0]
    # growth over the two halves of the fit window, as a check of exponential growth
    mid = 0.8 * t[-1]
    g1 = np.polyfit(t[late & (t < mid)], np.log(amax[late & (t < mid)]), 1)[0]
    g2 = np.polyfit(t[late & (t >= mid)], np.log(amax[late & (t >= mid)]), 1)[0]
    numbers = dict(gamma=float(g), gamma_halves=[float(g1), float(g2)], omega=float(abs(w)), t0=t0,
                   gamma_per_s=float(g / t0), omega_rad_s=float(abs(w) / t0), f_kHz=float(abs(w) / t0 / 2 / np.pi / 1e3),
                   rho_peak=float(rho[k]), t_end=float(t[-1]), fit_window=[float(t[late][0]), float(t[-1])])

    # 1. amplitude and fit
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogy(t, amax, color=C1, label=r"$\max_\rho |\Phi_6|$")
    ax.semilogy(t[late], np.exp(b + g * t[late]), color=C2, ls="--", lw=1.6,
                label=rf"fit, $\gamma = {g:.4f}\,\omega_{{A0}}$")
    ax.set(xlabel=r"$t\ [L_0/v_{A0}]$", ylabel=r"$|\Phi_{n=6}|$ [code units]", title="n = 6 amplitude")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "amplitude.png", dpi=150)
    plt.close(fig)

    # 2. radial structure at the diagnostic plane (theta index grid.attrs.diag_y_index), at a few times
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for tt, c, ls in ((0.25 * t[-1], C3, ":"), (0.6 * t[-1], C2, "--"), (t[-1], C1, "-")):
        i = int(np.argmin(np.abs(t - tt)))
        ax.plot(rho, A[i] / A[i].max(), color=c, ls=ls, label=rf"$t = {t[i]:.0f}\,t_0$")
    ax.set(xlabel=r"$\rho = \sqrt{\psi_t/\psi_{t,a}}$", ylabel=r"$|\Phi_6|$ (normalized)",
           title=r"Radial profile of $|\Phi_6|$ on the outboard midplane")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "radial.png", dpi=150)
    plt.close(fig)

    # 3. poloidal harmonics and poloidal-plane structure of the final Phi. The stored field is on (y, x, z) =
    # (theta_PEST, rho, phi) at cell centres; the n = 6 component is one wavelength over the wedge, and the 32 points
    # in theta resolve |m| <= 16, so both transforms are plain FFTs.
    if "final/Phi" in f:
        P = f["final/Phi"][()]
        y, zg = f["grid/y"][()], f["grid/z"][()]
        ny, nz = P.shape[0], P.shape[2]
        c = (2.0 / nz) * np.einsum("yxz,z->yx", P, np.exp(-6j * zg))          # Phi = Re[c exp(6 i phi)]
        a = np.fft.fft(c, axis=0) / ny                                        # c(theta) = sum_m a_m exp(i m (theta - y0))
        m = np.fft.fftfreq(ny, 1.0 / ny).astype(int)
        amp = np.abs(a).max(1)
        top = [int(k) for k in np.argsort(-amp)[:4]]
        numbers["harmonics"] = {str(abs(int(m[k]))): float(amp[k] / amp.max()) for k in top}
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for k, col, ls in zip(sorted(top, key=lambda k: abs(m[k])), (C3, C1, C2, INK2), (":", "-", "--", "-.")):
            ax.plot(rho, np.abs(a[k]) / amp.max(), color=col, ls=ls, lw=2 if amp[k] > 0.1 * amp.max() else 1.4,
                    label=rf"$|m| = {abs(m[k])}$")
        ax.set(xlabel=r"$\rho$", ylabel=r"$|\Phi_{6,m}|$ (normalized)", title="Poloidal harmonics of $\\Phi_6$ at the end")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "harmonics.png", dpi=150)
        plt.close(fig)

        # poloidal plane at phi = 0: Fourier interpolation in theta of Phi, R and Z
        th = np.linspace(-np.pi, np.pi, 257)
        E = np.exp(1j * np.outer(m, th - y[0]))                              # (m, theta)
        half = np.abs(m) == ny // 2
        def interp(blockyx):                                                  # real data: Re keeps the Nyquist term
            b = np.fft.fft(blockyx, axis=0) / ny
            return np.real(np.einsum("mx,mt->tx", b, E))
        def interp_c(cyx):
            b = np.fft.fft(cyx, axis=0) / ny
            b[half] = 0
            return np.einsum("mx,mt->tx", b, E)
        Rb = f["equilibrium/blocks/R"][:, :, 0].T * L0                       # (y, x) in m
        Zb = f["equilibrium/blocks/Z"][:, :, 0].T * L0
        Rf, Zf = interp(Rb), interp(Zb)
        Pf = np.real(interp_c(c))                                             # Phi at phi = 0
        v = np.abs(Pf).max()
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.pcolormesh(Rf, Zf, Pf / v, cmap=DIVERGING, vmin=-1, vmax=1, shading="gouraud", rasterized=True)
        for r0 in (0, -1):                                                    # inner and outer radial boundaries
            ax.plot(Rf[:, r0], Zf[:, r0], color=INK2, lw=1.0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set(xlabel="R [m]", ylabel="Z [m]", title=r"$\Phi_6$ at $\varphi = 0$, end of the run")
        fig.colorbar(im, ax=ax, shrink=0.8, label="normalized")
        fig.tight_layout()
        fig.savefig(FIG / "mode.png", dpi=150)
        plt.close(fig)

    # 4. fast-ion loading
    if "initial/Alpha/density" in f:
        d = f["initial/Alpha/density"][()]
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.plot(rho, d / 1e17, "o", ms=3.5, color=C1, label="markers")
        r = np.linspace(0, 1, 200)
        ax.plot(r, itpa_fast_density(r) / 1e17, color=C2, label="ITPA formula")
        ax.set(xlabel=r"$\rho = \sqrt{s}$", ylabel=r"$n_h\ [10^{17}\,\mathrm{m^{-3}}]$",
               title="Fast-ion density loaded in the alpha slot")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "loading.png", dpi=150)
        plt.close(fig)
        numbers["loading_max_rel_error"] = float(np.abs(d / itpa_fast_density(rho) - 1).max())
    (FIG / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "TCV/ITPA_ref/ITPA_ref.h5")
