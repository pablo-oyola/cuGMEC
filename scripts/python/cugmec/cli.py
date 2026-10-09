"""
cugmec command line.

From a gEQDSK and kinetic profiles to a case ready to build (PEST coordinates of the g-file, via pyCOCOS):
    cugmec from-geqdsk g123456.01000 profiles.h5 -o eq.h5 --ion-mass 2 --types 1 2 3 --case runs/x --template src/cuGMEC_param.h
From a DESC equilibrium, or from standard2D.mat and NTP.mat:
    cugmec from-desc desc_eq.h5 profiles.h5 -o eq.h5 --ion-mass 2 --types 1 2 3
    cugmec prepare standard2D.mat NTP.mat -o eq.h5 --ion-mass 1.0 --types 1 1 3
Then:
    cugmec case eq.h5 runs/x --template src/cuGMEC_param.h --set dt=0.05
    make CASE=runs/x && mpirun -n 1 runs/x/cuGMEC
    cugmec outputs runs/x/output -o runs/x/x.h5
"""
import argparse
import json
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


def _fit(items):
    fit = {}
    for item in items:
        name, spec = item.split("=")
        window, order = spec.split(",")
        fit[name] = (int(window), int(order))
    return fit


def _species_inputs(a, fit):
    return {"ion_mass": a.ion_mass, "ion_type": a.types[0], "alpha_type": a.types[1], "beam_type": a.types[2], "fit": fit}


def _finish(a, S, P, inputs):
    """build_2d, write the equilibrium HDF5 and, with --case, the case directory."""
    fit = _fit(a.fit)
    blocks, norm = eq.build_2d(S, P, a.ion_mass, *a.types, fit=fit)
    inputs = {**_species_inputs(a, fit), **inputs}
    h5io.save_equilibrium(a.output, blocks, norm, inputs=inputs, profiles=P)
    print(f"wrote {a.output} ({norm['gridNx']} x {norm['gridNy']})")
    if a.case:
        if not a.template:
            sys.exit("--case needs --template")
        make_case(a.output, a.case, a.template, a.set)


def cmd_prepare(a):
    S, P = _load_mat(a.standard2D), _load_mat(a.ntp)
    _finish(a, S, P, {"standard2D": str(Path(a.standard2D).resolve()), "standard2D_sha256": h5io.sha256(a.standard2D),
                      "ntp": str(Path(a.ntp).resolve()), "ntp_sha256": h5io.sha256(a.ntp)})


def _profile_inputs(a, coord):
    return {"profiles": str(Path(a.profiles).resolve()), "profiles_sha256": h5io.sha256(a.profiles), "profiles_coordinate": coord}


def cmd_from_desc(a):
    import desc
    from desc.equilibrium import EquilibriaFamily
    from desc.io import load

    from . import desc_eval, profiles

    coord, x, prof = profiles.read_profiles(a.profiles)
    if coord != "rho":
        sys.exit(f"from-desc needs profiles on DESC rho (found {coord}); use from-geqdsk for psi_n, rho_pol or rho_tor")
    P = profiles.to_ntp(x, prof, a.types)
    equil = load(a.desc)
    if isinstance(equil, EquilibriaFamily):  # a continuation run keeps the final equilibrium last
        equil = equil[-1]
    S = desc_eval.standard2d(equil, a.rho[0], a.rho[1], a.grid[0], a.grid[1])
    _finish(a, S, P, {"desc_equilibrium": str(Path(a.desc).resolve()), "desc_sha256": h5io.sha256(a.desc),
                      "desc_version": desc.__version__, **_profile_inputs(a, coord)})


def cmd_from_geqdsk(a):
    from . import profiles
    from .geqdsk import GEqdsk

    g = GEqdsk(a.geqdsk, cocos=a.cocos)
    if g.flipped:
        print("note: psirz has the opposite sign to simag/sibry in this file; a corrected copy is used", file=sys.stderr)
    if a.method == "pycocos":
        from .pest import PestMap
        pest_map = PestMap(g, a.psin_b, spectral_max_mode=a.modes)
        rho_of, psin_of = pest_map.rho, pest_map.psin_of_rho  # consistent with the flux map's own q
    else:
        rho_of, psin_of = (lambda x: g.rho(x, a.psin_b)), (lambda r: g.psin_of_rho(r, a.psin_b))

    coord, x, prof = profiles.read_profiles(a.profiles)
    rho = x if coord == "rho" else rho_of(g.to_psin(coord, x))
    P = profiles.to_ntp(rho, prof, a.types)
    kinetic = profiles.kinetic_pressure(P, a.types)
    p_axis = {"geqdsk": float(g.pressure(0.0)), "pprime": float(g.pressure_from_pprime(0.0)), "kinetic": float(kinetic(0.0))}
    print("pressure on axis [Pa]: " + ", ".join(f"{k} {v:.4g}" for k, v in p_axis.items()) + f"; using {a.pressure}")
    pressure = {"geqdsk": None, "kinetic": kinetic, "pprime": lambda r: g.pressure_from_pprime(psin_of(r))}[a.pressure]
    inputs = {"geqdsk": str(Path(a.geqdsk).resolve()), "geqdsk_sha256": h5io.sha256(a.geqdsk), "method": a.method,
              "pressure_source": a.pressure, "pressure_on_axis": p_axis, **_profile_inputs(a, coord)}

    if a.method == "pycocos":
        S, report = pest_map.standard2d(a.rho[0], a.rho[1], a.grid[0], a.grid[1], pressure)
        print("checks: " + json.dumps(report))
        inputs["pest_checks"] = report
    else:
        import desc

        from . import desc_build, desc_eval

        equil, report = desc_build.build_desc(g, a.psin_b, a.M, a.iota_order, a.pressure_order, pressure, a.solver, a.verbose)
        desc_out = a.desc_out or str(Path(a.output).with_name(Path(a.output).stem + "_desc.h5"))
        equil.save(desc_out)
        checks = desc_build.compare(equil, g, a.psin_b)
        print(f"wrote {desc_out}")
        print("fit:    " + json.dumps({k: v for k, v in report.items() if "error" in k}))
        print("checks: " + json.dumps(checks))
        S = desc_eval.standard2d(equil, a.rho[0], a.rho[1], a.grid[0], a.grid[1])
        inputs.update({"desc_equilibrium": str(Path(desc_out).resolve()), "desc_version": desc.__version__,
                       "desc_build": report, "desc_checks": checks})
    _finish(a, S, P, inputs)


def make_case(equilibrium, case_dir, template, sets=()):
    """Case directory from an equilibrium HDF5 file: cuGMEC_param.h (template + equilibrium + paths) and the binary."""
    blocks, norm, info = h5io.load_equilibrium(equilibrium)
    case = Path(case_dir).resolve()
    bin_name = f"MHDEquilibrium_{norm['gridNx']}_{norm['gridNy']}.bin"
    values = {k: norm[k] for k in EQUILIBRIUM_KEYS}
    values.update(inputDir=str(case), outputDir=str(case / "output"), MHDEquilibrium=bin_name)
    inputs = info["inputs"]
    if "ion_mass" in inputs:
        values["IonMass"] = float(inputs["ion_mass"])  # VA0, Va and Rho in the file were computed with it
    for item in sets:
        name, _, text = item.partition("=")
        if name in values and name not in ("inputDir", "outputDir", "MHDEquilibrium"):
            print(f"warning: --set {name} overrides a value that comes from the equilibrium", file=sys.stderr)
        values[name] = _value(text)

    text = params.patch_param_h(Path(template).read_text(), values)
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
    print(f"next: make CASE={case_dir}   (run from the repository root)")


def cmd_case(a):
    make_case(a.equilibrium, a.case_dir, a.template, a.set)


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


def _species_options(p):
    p.add_argument("--ion-mass", type=float, required=True, help="thermal ion mass / m_p")
    p.add_argument("--types", type=int, nargs=3, default=[eq.MAXWELL, eq.SLOWING, eq.SLOWING], metavar=("ION", "ALPHA", "BEAM"),
                   help="1 Maxwellian, 2 slowing-down, 3 absent (codes of generateInput2D.m)")
    p.add_argument("--fit", action="append", default=[], metavar="NAME=WINDOW,ORDER",
                   help="profile fit, e.g. --fit nb=22,6 (names: ne Te ni Ti Pi na Ta Pa nb Tb Pb)")


def _grid_options(p):
    p.add_argument("--rho", type=float, nargs=2, default=[0.08, 0.90], metavar=("RHO0", "RHO1"), help="radial domain in DESC rho")
    p.add_argument("--grid", type=int, nargs=2, default=[256, 32], metavar=("NX", "NY"), help="gridNx (multiple of 128) and gridNy")


def _case_options(p):
    p.add_argument("--case", help="also create this case directory (see `cugmec case`)")
    p.add_argument("--template", help="cuGMEC_param.h to start the case from")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="change a constant of the template")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cugmec", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("from-geqdsk", help="gEQDSK + kinetic profiles -> equilibrium HDF5 (PEST coordinates from pyCOCOS, or a DESC solve)")
    p.add_argument("geqdsk")
    p.add_argument("profiles", help="kinetic profiles (.h5, NTP-style .mat, or text table); see cugmec/profiles.py")
    p.add_argument("-o", "--output", required=True, help="equilibrium HDF5 file to write")
    _species_options(p)
    _grid_options(p)
    p.add_argument("--method", choices=["pycocos", "desc"], default="pycocos",
                   help="pycocos: PEST coordinates of the g-file's own flux map (default); desc: re-solve with DESC")
    p.add_argument("--psin-b", type=float, default=0.995, help="g-file psi_n where rho = 1 (< 1 for diverted plasmas)")
    p.add_argument("--modes", type=int, default=32, help="pycocos: Fourier modes of the flux surfaces")
    p.add_argument("--cocos", type=int, default=1, help="COCOS of the g-file; only >= 11 matters (psi in Wb, not Wb/rad)")
    p.add_argument("--pressure", choices=["geqdsk", "pprime", "kinetic"], default="geqdsk",
                   help="pressure given to DESC: the g-file pres, the integral of its pprime, or the sum of the kinetic profiles")
    p.add_argument("--M", type=int, default=16, help="desc: DESC poloidal resolution (also used to fit the boundary)")
    p.add_argument("--iota-order", default="auto", help="desc: even power-series order of iota(rho); auto: lowest order fitting q to 0.1%%")
    p.add_argument("--pressure-order", default="auto", help="desc: even power-series order of p(rho); auto: lowest order fitting p to 0.1%%")
    p.add_argument("--solver", choices=["auto", "direct", "continuation"], default="auto",
                   help="desc: direct eq.solve as in compute2D_0170.ipynb; auto falls back to DESC's continuation if it fails")
    p.add_argument("--desc-out", help="desc: where to save the DESC equilibrium (default: <output>_desc.h5)")
    p.add_argument("--verbose", type=int, default=1)
    _case_options(p)
    p.set_defaults(func=cmd_from_geqdsk)

    p = sub.add_parser("from-desc", help="DESC equilibrium + kinetic profiles on DESC rho -> equilibrium HDF5")
    p.add_argument("desc", help="DESC equilibrium file (.h5)")
    p.add_argument("profiles")
    p.add_argument("-o", "--output", required=True)
    _species_options(p)
    _grid_options(p)
    _case_options(p)
    p.set_defaults(func=cmd_from_desc)

    p = sub.add_parser("prepare", help="standard2D.mat + NTP.mat -> equilibrium HDF5")
    p.add_argument("standard2D")
    p.add_argument("ntp")
    p.add_argument("-o", "--output", required=True, help="equilibrium HDF5 file to write")
    _species_options(p)
    _case_options(p)
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
