"""
cugmec command line.

    cugmec prepare standard2D.mat NTP.mat -o eq.h5 --ion-mass 1.0 --types 1 1 3
    cugmec case eq.h5 runs/itpa --template src/cuGMEC_param.h --set dt=0.05
    make CASE=runs/itpa && mpirun -n 1 runs/itpa/cuGMEC
    cugmec outputs runs/itpa/output -o runs/itpa/itpa.h5
"""
import argparse
import sys
from pathlib import Path

from scipy.io import loadmat

from . import equilibrium as eq
from . import h5io, params

# cuGMEC_param.h constants that come from the equilibrium
EQUILIBRIUM_KEYS = ("B0", "L0", "VA0", "RHO0", "RHO1", "PSITMAX", "IonBeta", "AlphaBeta", "BeamBeta", "gridNx", "gridNy", "NFP")


def _load_mat(path):
    return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}


def _value(text):
    """Command-line value for --set: true/false, int, float, or string."""
    low = text.lower()
    if low in ("true", "truetype"):
        return True
    if low in ("false", "falsetype"):
        return False
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def cmd_prepare(a):
    fit = {}
    for item in a.fit:
        name, spec = item.split("=")
        window, order = spec.split(",")
        fit[name] = (int(window), int(order))
    S, P = _load_mat(a.standard2D), _load_mat(a.ntp)
    blocks, norm = eq.build_2d(S, P, a.ion_mass, *a.types, fit=fit)
    inputs = {"ion_mass": a.ion_mass, "ion_type": a.types[0], "alpha_type": a.types[1], "beam_type": a.types[2],
              "fit": fit, "standard2D": str(Path(a.standard2D).resolve()), "standard2D_sha256": h5io.sha256(a.standard2D),
              "ntp": str(Path(a.ntp).resolve()), "ntp_sha256": h5io.sha256(a.ntp)}
    h5io.save_equilibrium(a.output, blocks, norm, inputs=inputs, profiles=P)
    print(f"wrote {a.output} ({norm['gridNx']} x {norm['gridNy']})")


def cmd_case(a):
    blocks, norm, info = h5io.load_equilibrium(a.equilibrium)
    case = Path(a.case_dir).resolve()
    bin_name = f"MHDEquilibrium_{norm['gridNx']}_{norm['gridNy']}.bin"
    values = {k: norm[k] for k in EQUILIBRIUM_KEYS}
    values.update(inputDir=str(case), outputDir=str(case / "output"), MHDEquilibrium=bin_name)
    inputs = info["inputs"]
    if "ion_mass" in inputs:
        values["IonMass"] = float(inputs["ion_mass"])  # VA0, Va and Rho in the file were computed with it
    for item in a.set:
        name, _, text = item.partition("=")
        if name in values and name not in ("inputDir", "outputDir", "MHDEquilibrium"):
            print(f"warning: --set {name} overrides a value that comes from the equilibrium", file=sys.stderr)
        values[name] = _value(text)

    text = params.patch_param_h(Path(a.template).read_text(), values)
    p = params.parse_param_h(text)

    errors, warnings = [], params.check_grid(p)
    codes = [inputs.get(k) for k in ("ion_type", "alpha_type", "beam_type")]
    for species, code in zip(("Ion", "Alpha", "Beam"), codes):
        enabled, dist = p[f"if{species}"], p[f"{species}Type"]
        if code is None:
            continue
        if code == eq.ABSENT and enabled:
            errors.append(f"if{species} is trueType but the equilibrium has no {species} profiles")
        elif code != eq.ABSENT and not enabled:
            warnings.append(f"the equilibrium has {species} profiles but if{species} is falseType")
        elif code == eq.MAXWELL and dist != "Maxwell":
            errors.append(f"{species}Type is {dist}, but the equilibrium stores a Maxwellian (T in keV)")
        elif code == eq.SLOWING and not str(dist).startswith("Slowing"):
            errors.append(f"{species}Type is {dist}, but the equilibrium stores a slowing-down distribution (v_c / VA0)")
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if errors:
        sys.exit("cannot create the case:\n  " + "\n  ".join(errors) + "\nfix the template or use --set")

    (case / "output").mkdir(parents=True, exist_ok=True)
    eq.write_bin(case / bin_name, blocks, norm["gridNx"], norm["gridNy"], info["grid"]["equilNz"])
    (case / "cuGMEC_param.h").write_text(text)
    print(f"wrote {case / 'cuGMEC_param.h'} and {case / bin_name}")
    print(f"next: make CASE={a.case_dir}   (run from the repository root)")


def cmd_export_bin(a):
    norm, _ = h5io.export_bin(a.equilibrium, a.output)
    print(f"wrote {a.output} ({norm['gridNx']} x {norm['gridNy']})")


def cmd_import_bin(a):
    p = params.parse_param_h(Path(a.param).read_text())
    equil_nz = 1 if p["NFP"] == 1 else p["gridNz"]
    h5io.import_bin(a.binary, a.output, p["gridNx"], p["gridNy"], equil_nz, norm=p, NFP=p["NFP"])
    print(f"wrote {a.output}")


def cmd_outputs(a):
    from .outputs import convert_outputs
    report = convert_outputs(a.output_dir, a.output, param_path=a.param, restart=a.restart)
    for line in report:
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cugmec", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="standard2D.mat + NTP.mat -> equilibrium HDF5")
    p.add_argument("standard2D")
    p.add_argument("ntp")
    p.add_argument("-o", "--output", required=True, help="equilibrium HDF5 file to write")
    p.add_argument("--ion-mass", type=float, required=True, help="thermal ion mass / m_p")
    p.add_argument("--types", type=int, nargs=3, default=[eq.MAXWELL, eq.SLOWING, eq.SLOWING], metavar=("ION", "ALPHA", "BEAM"),
                   help="1 Maxwellian, 2 slowing-down, 3 absent (codes of generateInput2D.m)")
    p.add_argument("--fit", action="append", default=[], metavar="NAME=WINDOW,ORDER",
                   help="profile fit, e.g. --fit nb=22,6 (names: ne Te ni Ti Pi na Ta Pa nb Tb Pb)")
    p.set_defaults(func=cmd_prepare)

    c = sub.add_parser("case", help="equilibrium HDF5 -> case directory with cuGMEC_param.h and the binary")
    c.add_argument("equilibrium")
    c.add_argument("case_dir")
    c.add_argument("--template", required=True, help="cuGMEC_param.h with the physics and run settings to start from")
    c.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="change any constant or switch of the template, e.g. --set dt=0.05 --set ifOutputPhi=true")
    c.set_defaults(func=cmd_case)

    e = sub.add_parser("export-bin", help="equilibrium HDF5 -> the binary cuGMEC reads")
    e.add_argument("equilibrium")
    e.add_argument("-o", "--output", required=True)
    e.set_defaults(func=cmd_export_bin)

    i = sub.add_parser("import-bin", help="existing equilibrium binary -> HDF5 (grid and normalization from param.h)")
    i.add_argument("binary")
    i.add_argument("--param", required=True, help="the cuGMEC_param.h the binary was made for")
    i.add_argument("-o", "--output", required=True)
    i.set_defaults(func=cmd_import_bin)

    o = sub.add_parser("outputs", help="cuGMEC output directory -> HDF5")
    o.add_argument("output_dir", help="the outputDir of the run (contains initial/ and final/)")
    o.add_argument("-o", "--output", required=True)
    o.add_argument("--param", help="cuGMEC_param.h of the run (default: <output_dir>/final/cuGMEC_param.h)")
    o.add_argument("--restart", action="store_true", help="also store MHDContinue and PICContinue (particles; can be large)")
    o.set_defaults(func=cmd_outputs)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
