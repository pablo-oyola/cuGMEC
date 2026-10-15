"""
cuGMEC output directory -> one HDF5 file.

Shapes, dtypes and switches come from the run's own cuGMEC_param.h (outputDir/final/cuGMEC_param.h, written by the
code). Layout (format_version 1):

    /                       attrs: format, format_version, created, generator, source
    /param                  every constant and switch of cuGMEC_param.h as attributes; its text in /param/cuGMEC_param.h
    /grid                   x, rho, y, z, mode (FFT index), n (toroidal mode number = tubes * mode)
    /time                   step_diag, t_diag, step_output, t_output; t in units of L0/VA0
    /equilibrium            the equilibrium the run read (same layout as an equilibrium HDF5 file)
    /diagnostics/<name>     1D diagnostics, one record every diagSteps
    /fields/<name>          total<name>.bin, one record every outputSteps: (time_output, y, x, z)
    /final/<name>           state at the last step: (y, x, z)
    /initial/<Sp>/          initial radial density and pressure of each species
    /phase_space/<Sp>/      axes E, Pphi, Lambda; f0, jacobian, orbit, delta_f, power
    /pitch_space/<Sp>/      axes vpara, vperp; f0, jacobian, delta_f, power
    /restart/               MHDContinue and PICContinue, only when requested

Every dataset has the attributes dims, description and source, and unit when it is known.
"""
from pathlib import Path

import h5py
import numpy as np

from .equilibrium import GHOST, MP, QE, read_bin
from .h5io import NORMALIZATION, stamp, write_equilibrium
from .params import parse_param_h

FORMAT = "cuGMEC output"
FORMAT_VERSION = 1
SPECIES = ("Ion", "Alpha", "Beam")
TOTALS = ("Phi", "A", "dNe", "dTe", "dP", "dPi", "dPa", "dPb")
SNAPSHOTS = ("w", "A", "dNe", "dTe", "dP", "Phi", "dJpB", "dPe", "dPi", "dPa", "dPb", "dNi", "dNa", "dNb")
# final/<Sp>Loss.bin columns, in the order of LossVar (src/cuGMEC_model.h): count, delta f content w and energy w E of
# the markers reset at the outer boundary, their f0 content p and energy p E; count and energies at the inner boundary
LOSS_COLUMNS = ("out_N", "out_w", "out_wE", "out_p", "out_pE", "in_N", "in_wE", "in_pE")
CONTINUE_FIELDS = ("Phi", "A", "dNe", "dTe", "dP", "dPi", "dPa", "dPb", "dNi", "dNa", "dNb")
MARKER_COMPONENTS = ("x", "y", "z", "vpara", "w", "p", "mu")
ORBIT_COLUMNS = ("id", "orbit_type", "dtheta", "dphi_total", "dphi_vpara", "dT", "E", "Pphi", "Lambda")
ORBIT_TYPES = "0.5 pad, 1.5 lost, 2.5 co-passing, 3.5 counter-passing, 4.5 trapped, 5.5 unknown"


class _Writer:
    def __init__(self, h5, root):
        self.h5, self.root, self.report = h5, root, []

    def put(self, rel, path, dtype, shape, dims, description, unit=None, enabled=True):
        """Store root/rel as dataset `path`. Skips files whose switch is off or whose size does not match."""
        f = self.root / rel
        if not enabled:
            if f.exists():
                self.report.append(f"ignored  {rel} (its switch is off in cuGMEC_param.h)")
            return None
        if not f.exists():
            self.report.append(f"missing  {rel}")
            return None
        data = np.fromfile(f, dtype=dtype)
        known = int(np.prod([n for n in shape if n != -1]))
        if (data.size % known if -1 in shape else data.size != known):
            self.report.append(f"skipped  {rel}: {data.size} values do not fit shape {tuple(shape)}")
            return None
        return self.dataset(path, data.reshape(shape), dims, description, unit, source=rel)

    def dataset(self, path, data, dims, description, unit=None, source=None):
        opts = dict(compression="gzip", compression_opts=4, shuffle=True) if data.nbytes > (1 << 20) else {}
        ds = self.h5.create_dataset(path, data=data, **opts)
        ds.attrs.update(dims=",".join(dims), description=description)
        if unit:
            ds.attrs["unit"] = unit
        if source:
            ds.attrs["source"] = source
            self.report.append(f"stored   {source} -> {path} {data.shape}")
        return ds


def diag_y_index(p):
    """Global y index of the plane sampled by the 1D diagnostics (cuGMEC_setup.h diagY and the sampling device)."""
    hosts, devs, ny = p["hostNums"], p["devNums"], p["gridNy"]
    if hosts * devs == 1:
        return ny // 2
    if hosts == 1:
        return (devs // 2) * (ny // devs)
    return (hosts // 2) * (ny // hosts)


def phase_space_axes(p, eq, species):
    """E, Pphi, Lambda node coordinates as computed by computePhaseSpaceRange (src/cuGMEC_model.h)."""
    mass, char, vmin, vmax = (float(p[species + k]) for k in ("Mass", "Char", "Vmin", "Vmax"))
    nx, ny = p["gridNx"], p["gridNy"]
    q, psip, J, B = (eq[k] for k in ("q", "psip", "J", "B"))
    cov = eq["SFAcovyz"][:, GHOST:GHOST + ny, :]
    if p["NFP"] != 1:
        cov = cov + q * eq["SFAcovzz"][:, GHOST:GHOST + ny, :]
    drho = p["RHO1"] - p["RHO0"]
    psitmax = p["PSITMAX"] / (p["B0"] * p["L0"] ** 2)
    cm = p["VA0"] / (p["L0"] * (QE * p["B0"] / MP))
    rho = (p["RHO0"] + np.arange(nx) / (nx - 1) * drho)[:, None, None]
    pphi = np.concatenate([cm * mass * v * 2 * psitmax * drho * rho * cov / (q * J * B) - char * psip for v in (-vmax, vmax)])
    lo = np.array([0.5 * mass * vmin**2, pphi.min(), 0.0])
    hi = np.array([0.5 * mass * vmax**2, pphi.max(), (1.0 / B).max()])
    real = np.float32 if p["picReal"] == "float" else np.float64  # the code keeps the range in picReal
    lo, hi = lo.astype(real).astype(float), hi.astype(real).astype(float)
    return [np.linspace(lo[i], hi[i], n) for i, n in enumerate((p["gridE"], p["gridPphi"], p["gridLambda"]))]


def convert_outputs(output_dir, h5_path, param_path=None, restart=False):
    """Convert every output of one run into `h5_path`. Returns a report, one line per file."""
    root = Path(output_dir)
    param_path = Path(param_path) if param_path else root / "final" / "cuGMEC_param.h"
    if not param_path.exists():
        raise FileNotFoundError(f"{param_path} not found; pass the run's cuGMEC_param.h explicitly")
    text = param_path.read_text()
    p = parse_param_h(text)

    nx, ny, nz = p["gridNx"], p["gridNy"], p["gridNz"]
    equil_nz = 1 if p["NFP"] == 1 else nz
    n_modes = p["rightN"] - p["leftN"] + 1
    n_diag = p["totalSteps"] // p["diagSteps"] + 1
    n_out = p["totalSteps"] // p["outputSteps"] + 1
    mhd = "<f8" if p["mhdReal"] == "double" else "<f4"
    pic = "<f8" if p["picReal"] == "double" else "<f4"
    start = p["continueSteps"]
    first_run = not p["ifContinue"] or start == 0
    enabled = {s: p["if" + s] for s in SPECIES}

    with h5py.File(h5_path, "w") as h5:
        stamp(h5, FORMAT, FORMAT_VERSION)
        h5.attrs["source"] = str(root.resolve())
        w = _Writer(h5, root)

        g = h5.create_group("param")
        g.attrs.update({k: v for k, v in p.items() if v is not None})
        g["cuGMEC_param.h"] = text

        x = np.arange(nx) / (nx - 1)
        mode = np.arange(p["leftN"], p["rightN"] + 1)
        w.dataset("grid/x", x, ["x"], "radial coordinate in [0, 1]")
        w.dataset("grid/rho", p["RHO0"] + x * (p["RHO1"] - p["RHO0"]), ["x"], "rho_tor")
        w.dataset("grid/y", -np.pi + (np.arange(ny) + 0.5) * 2 * np.pi / ny, ["y"], "theta_PEST (cell centres)", "rad")
        w.dataset("grid/z", -np.pi / p["tubes"] + (np.arange(nz) + 0.5) * 2 * np.pi / (p["tubes"] * nz), ["z"],
                  "toroidal coordinate over 1/tubes of the torus (cell centres)", "rad")
        w.dataset("grid/mode", mode, ["mode"], "index of the z Fourier mode")
        w.dataset("grid/n", p["tubes"] * mode, ["mode"], "toroidal mode number = tubes * mode")
        h5["grid"].attrs.update(diag_y_index=diag_y_index(p))

        tau = p["L0"] / p["VA0"]
        for kind, n, every in (("diag", n_diag, p["diagSteps"]), ("output", n_out, p["outputSteps"])):
            step = start + np.arange(n) * every
            w.dataset(f"time/step_{kind}", step, [f"time_{kind}"], f"time step of each {kind} record")
            w.dataset(f"time/t_{kind}", step * p["dt"], [f"time_{kind}"], f"time of each {kind} record", "L0/VA0")
        h5["time"].attrs.update(seconds_per_unit=tau)

        # ---- equilibrium the run read (copied by the code into final/)
        eq = None
        eq_file = root / "final" / p["MHDEquilibrium"]
        if eq_file.exists():
            try:
                eq = read_bin(eq_file, nx, ny, equil_nz)
                norm = {k: p[k] for k in NORMALIZATION}
                norm.update(gridNx=nx, gridNy=ny, NFP=p["NFP"])
                write_equilibrium(h5.create_group("equilibrium"), eq, norm, equil_nz,
                                  inputs={"source": f"final/{p['MHDEquilibrium']}"})
                w.report.append(f"stored   final/{p['MHDEquilibrium']} -> equilibrium/")
            except ValueError as err:
                w.report.append(f"skipped  final/{p['MHDEquilibrium']}: {err}")
        else:
            w.report.append(f"missing  final/{p['MHDEquilibrium']}")

        # ---- MHD diagnostics
        d_mode, d_line = (n_diag, nx, n_modes), (n_diag, nx)
        dims_mode, dims_line = ["time_diag", "x", "mode"], ["time_diag", "x"]
        on = p["ifDiagAmplitude"]
        w.put("final/amplitude.bin", "diagnostics/amplitude", mhd, d_mode, dims_mode,
              "|FFT_z(Phi)| / gridNz per mode at y = grid.diag_y_index (x2 except mode 0 and Nyquist)", enabled=on)
        w.put("final/RealMode.bin", "diagnostics/RealMode", mhd, d_mode, dims_mode,
              "real part of the unnormalized FFT_z(Phi) at y = grid.diag_y_index", enabled=on)
        w.put("final/ImagMode.bin", "diagnostics/ImagMode", mhd, d_mode, dims_mode,
              "imaginary part of the unnormalized FFT_z(Phi) at y = grid.diag_y_index", enabled=on)
        w.put("final/frequency.bin", "diagnostics/frequency", mhd, d_line, dims_line,
              "Phi at y = grid.diag_y_index, z index 0 (time signal for frequency analysis)", enabled=p["ifDiagFrequency"])
        w.put("final/shearing.bin", "diagnostics/shearing", mhd, d_line, dims_line,
              "Hahm-Burrell shearing rate from the (n, m) = (0, 0) part of Phi", "1/s", enabled=p["ifDiagShearing"])
        for name, text_ in (("Epara", "-E_parallel"), ("EparaES", "electrostatic part of -E_parallel")):
            w.put(f"final/{name}.bin", f"diagnostics/{name}", mhd, d_line, dims_line,
                  f"{text_} at y = grid.diag_y_index, z index 0", enabled=p["ifDiagEparallel"])
        for name, text_ in (("MaxwellDrive", "Maxwell-stress zonal drive / dt"), ("ReynoldsDrive", "Reynolds-stress zonal drive / dt"),
                            ("ZonalDrive", "(w_end - w_begin) / dt of the zonal vorticity")):
            w.put(f"final/{name}.bin", f"diagnostics/{name}", mhd, d_line, dims_line,
                  f"{text_} at y = grid.diag_y_index, z index 0", enabled=p["ifDiagZFDrive"])
        for s in SPECIES:
            w.put(f"final/{s}Density.bin", f"diagnostics/{s}Density", mhd, d_line, dims_line,
                  f"radial delta n profile of {s} (sum over all GPUs)", enabled=p["ifDiagDensity"] and enabled[s])
            w.put(f"final/{s}Diffusivity.bin", f"diagnostics/{s}Diffusivity", mhd, (n_diag, n_modes, nx),
                  ["time_diag", "mode", "x"], f"radial diffusivity of {s} per toroidal mode",
                  enabled=p["ifDiagDiffusivity"] and enabled[s])
            ds = w.put(f"final/{s}Loss.bin", f"diagnostics/{s}Loss", np.float64, (n_diag, len(LOSS_COLUMNS)),
                       ["time_diag", "column"],
                       f"{s} markers reset at the radial boundaries since the start of the run segment (cumulative); "
                       "w and p carry partConst, E = m v_par^2/2 + mu B in code units", enabled=enabled[s])
            if ds is not None:
                ds.attrs["columns"] = ",".join(LOSS_COLUMNS)

        # ---- MHD fields
        for name in TOTALS:
            w.put(f"final/total{name}.bin", f"fields/{name}", mhd, (n_out, ny, nx, nz), ["time_output", "y", "x", "z"],
                  f"{name} every outputSteps (y ghost cells removed)", enabled=p["ifOutput" + name])
        for name in SNAPSHOTS:
            w.put(f"final/{name}.bin", f"final/{name}", mhd, (ny, nx, nz), ["y", "x", "z"], f"{name} at the last step")

        # ---- PIC: initial profiles, phase space and pitch space
        for s in SPECIES:
            if not enabled[s]:
                continue
            w.put(f"initial/{s}Density.bin", f"initial/{s}/density", pic, (nx,), ["x"],
                  f"equilibrium density of {s} from the loaded markers (average over y, z)", "m^-3", enabled=first_run)
            w.put(f"initial/{s}Pressure.bin", f"initial/{s}/pressure", pic, (nx,), ["x"],
                  f"equilibrium pressure of {s} from the loaded markers (average over y, z)", "Pa", enabled=first_run)

            ph = f"phase_space/{s}"
            shape = (p["gridE"], p["gridPphi"], p["gridLambda"])
            dims = ["E", "Pphi", "Lambda"]
            any_phase = any(p[f"ifOutputPhaseSpace{k}"] for k in ("Jacobian", "Orbit", "F0", "DeltaF", "Power"))
            if any_phase and eq is not None:
                for axis, values, unit, text_ in zip(dims, phase_space_axes(p, eq, s), ("MP VA0^2", "normalized", "1/B0"),
                                                     ("energy", "canonical toroidal momentum", "mu / E")):
                    w.dataset(f"{ph}/{axis}", values, [axis], f"{text_} (node coordinates, recomputed as in the code)", unit)
            elif any_phase:
                w.report.append(f"note     {ph}: E/Pphi/Lambda axes not written (equilibrium file missing)")
            w.put(f"initial/{s}PhaseSpaceJacobian.bin", f"{ph}/jacobian", "<f8", shape, dims,
                  "number of uniformly loaded Monte Carlo markers per node", enabled=p["ifOutputPhaseSpaceJacobian"] and not p["ifContinue"])
            w.put(f"initial/{s}PhaseSpaceF0.bin", f"{ph}/f0", "<f8", shape, dims,
                  "node-deposited equilibrium distribution (same normalization as delta_f)",
                  enabled=p["ifOutputPhaseSpaceF0"] and not p["ifContinue"])
            orbit = w.put(f"initial/{s}PhaseSpaceOrbit.bin", f"{ph}/orbit", "<f8", (-1, len(ORBIT_COLUMNS)), ["record", "column"],
                          "orbit of each phase-space node; first half +vpara, second half -vpara",
                          enabled=p["ifOutputPhaseSpaceOrbit"] and not p["ifContinue"])
            if orbit is not None:
                orbit.attrs.update(columns=",".join(ORBIT_COLUMNS), orbit_type=ORBIT_TYPES)
            w.put(f"final/{s}PhaseDeltaF.bin", f"{ph}/delta_f", mhd, (n_out,) + shape, ["time_output"] + dims,
                  "node-deposited delta f", enabled=p["ifOutputPhaseSpaceDeltaF"])
            w.put(f"final/{s}PhasePower.bin", f"{ph}/power", mhd, (n_out, n_modes) + shape, ["time_output", "mode"] + dims,
                  "power from each toroidal mode to the particles", "W", enabled=p["ifOutputPhaseSpacePower"])

            pt = f"pitch_space/{s}"
            shape, dims = (p["gridVpara"], p["gridVperp"]), ["vpara", "vperp"]
            if any(p[f"ifOutputPitchSpace{k}"] for k in ("Jacobian", "F0", "DeltaF", "Power")):
                vmax = float(p[s + "Vmax"])
                w.dataset(f"{pt}/vpara", np.linspace(0.0 if s == "Beam" else -vmax, vmax, p["gridVpara"]), ["vpara"],
                          "parallel velocity (node coordinates)", "VA0")
                w.dataset(f"{pt}/vperp", np.linspace(0.0, vmax, p["gridVperp"]), ["vperp"],
                          "perpendicular velocity sqrt(2 mu B / m) (node coordinates)", "VA0")
            w.put(f"initial/{s}PitchSpaceJacobian.bin", f"{pt}/jacobian", "<f8", shape, dims,
                  "number of uniformly loaded Monte Carlo markers per node", enabled=p["ifOutputPitchSpaceJacobian"] and not p["ifContinue"])
            w.put(f"initial/{s}PitchSpaceF0.bin", f"{pt}/f0", "<f8", shape, dims,
                  "node-deposited equilibrium distribution (same normalization as delta_f)",
                  enabled=p["ifOutputPitchSpaceF0"] and not p["ifContinue"])
            w.put(f"final/{s}PitchDeltaF.bin", f"{pt}/delta_f", mhd, (n_out,) + shape, ["time_output"] + dims,
                  "node-deposited delta f", enabled=p["ifOutputPitchSpaceDeltaF"])
            w.put(f"final/{s}PitchPower.bin", f"{pt}/power", mhd, (n_out, n_modes) + shape, ["time_output", "mode"] + dims,
                  "power from each toroidal mode to the particles", "W", enabled=p["ifOutputPitchSpacePower"])

        if restart:
            _restart(w, p, mhd, pic, enabled, first_run)
    return w.report


def _restart(w, p, mhd, pic, enabled, first_run):
    nx, ny, nz = p["gridNx"], p["gridNy"], p["gridNz"]
    end = p["continueSteps"] + p["totalSteps"]
    for rel, path, on in ((f"final/MHDContinue_{end}.bin", "restart/final/mhd", True),
                          ("initial/MHDContinue_0.bin", "restart/initial/mhd", first_run)):
        ds = w.put(rel, path, mhd, (len(CONTINUE_FIELDS), ny, nx, nz), ["field", "y", "x", "z"],
                   "MHD restart fields", enabled=on)
        if ds is not None:
            ds.attrs["fields"] = ",".join(CONTINUE_FIELDS)

    hosts, devs, ppc = p["hostNums"], p["devNums"], p["ppcNums"]
    pic_host = nx * ny * nz // hosts * ppc
    pic_dev = nx * ny * nz // hosts // devs * ppc
    size = np.dtype(pic).itemsize
    species = [s for s in SPECIES if enabled[s]]
    block = size * (1 + 7 * pic_host) + 4 * pic_host
    for rel, group, on in ((f"final/PICContinue_{end}.bin", "restart/final", True), ("initial/PICContinue_0.bin", "restart/initial", first_run)):
        f = w.root / rel
        if not on or not species:
            continue
        if not f.exists():
            w.report.append(f"missing  {rel}")
            continue
        if f.stat().st_size != hosts * len(species) * block:
            w.report.append(f"skipped  {rel}: {f.stat().st_size} bytes, expected {hosts * len(species) * block}")
            continue
        raw = np.memmap(f, dtype=np.uint8, mode="r")
        for si, s in enumerate(species):
            const, keys, values = [], [], []
            for rank in range(hosts):
                off = (rank * len(species) + si) * block
                const.append(raw[off:off + size].view(pic)[0])
                off += size
                keys.append(raw[off:off + 4 * pic_host].view("<i4"))
                off += 4 * pic_host
                v = raw[off:off + size * 7 * pic_host].view(pic).reshape(devs, 7, pic_dev)
                values.append(v.transpose(1, 0, 2).reshape(7, pic_host))  # per GPU the 7 components are stored one after another
            values = np.concatenate(values, axis=1)
            g = f"{group}/{s}"
            for c, name in enumerate(MARKER_COMPONENTS):
                w.dataset(f"{g}/{name}", np.ascontiguousarray(values[c]), ["marker"], f"marker {name}")
            w.dataset(f"{g}/cell", np.concatenate(keys), ["marker"], "sorting key: cell index of each marker")
            w.dataset(f"{g}/const", np.asarray(const), ["rank"], "weight scale so that the pressure at x = 0 equals the beta")
        w.report.append(f"stored   {rel} -> {group}/<species> ({hosts * pic_host} markers per species)")
