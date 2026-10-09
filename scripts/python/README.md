# cugmec: Python tools for cuGMEC

The `cugmec` package prepares cuGMEC inputs and converts its outputs. HDF5 is the format you work with. The binary files cuGMEC reads and writes are generated from HDF5 or converted into it. It replaces the MATLAB stage of the 2D (tokamak) preprocessing: `generateInput2D.m`, i.e. `writeProfile1D`, `DESC2PEST2SFA2D`, `writeNormalization2D` and `writeEquilibrium2D`.

```bash
mamba env create -f scripts/python/environment.yml   # numpy, scipy, h5py, pytest
mamba activate cugmec
pip install -e scripts/python                        # adds the `cugmec` command; edits take effect without reinstalling
pip install -e /path/to/pyCOCOS                      # for from-geqdsk (default method)
pip install -e "scripts/python[desc]"                # DESC, for from-desc and from-geqdsk --method desc
```

numpy, scipy and h5py are enough for everything else.

## Workflow

Each case gets its own directory, its own `cuGMEC_param.h` and its own executable.

```bash
# 1. gEQDSK + kinetic profiles -> equilibrium HDF5 + case directory
cugmec from-geqdsk g204232.00650 profiles.txt -o d3d_eq.h5 --ion-mass 2 --types 1 2 3 \
       --case runs/d3d --template example/cuGMEC_param.h --set AlphaType=Slowing0 --set ifBeam=false
# 2. build and run (from the repository root)
make CASE=runs/d3d
mpirun -n 1 runs/d3d/cuGMEC
# 3. outputs -> HDF5
cugmec outputs runs/d3d/output -o runs/d3d/d3d.h5
```

Other ways in:

- `cugmec from-desc desc_eq.h5 profiles.h5 -o eq.h5 ...` starts from an existing DESC equilibrium, for example one made with `compute2D_0170.ipynb` or loaded from VMEC with `desc.vmec.VMECIO.load`. The profiles must then be on DESC ρ.
- `cugmec prepare standard2D.mat NTP.mat -o eq.h5 ...` starts from the files the notebooks write.
- `cugmec case eq.h5 runs/x --template ...` makes a case directory from any equilibrium HDF5 file.

### from-geqdsk

Two methods turn the g-file into the cuGMEC grid.

**`--method pycocos` (default).** This uses the g-file's own Grad–Shafranov solution, without re-solving it. It takes about 10 s.

1. [pyCOCOS](https://github.com/pablo-oyola/pyCOCOS) reads the file with its COCOS (`--cocos`). It traces the flux surfaces and builds straight-field-line (PEST) coordinates.
2. q comes from those surfaces, not from the file's `qpsi`. ρ, ψ_p, B^y and the q block therefore all match the flux map. EFIT's `qpsi` can be inconsistent with its own ψ(R, Z) by up to about 1%, and the difference is reported.
3. R and Z are sampled on a fine (ρ, θ) grid and fitted with a Fourier series in θ times Chebyshev series in ρ. That gives the metric and its derivatives, and √g. The Chebyshev degree is the lowest at which the fit residual stops improving, so derivatives do not amplify the noise of the traced surfaces.
4. Two quantities come from the field:
   - |B|, from B² = (F² + ψ_p′² g^ρρ)/R²;
   - J∥, from the Grad–Shafranov current, F′B + μ0 F p′/B.
5. θ = 0 lies on the outboard midplane ray, and θ runs as in DESC. ρ = 1 at ψ_N = `--psin-b`: use the default 0.995 for diverted plasmas, and 1.0 for limited ones.

A ψ map stored with the opposite sign to `simag` and `sibry` is detected, and a corrected copy is passed to pyCOCOS.

Checks, printed and stored under `inputs/pest_checks`:

| Check | Meaning |
|---|---|
| `geqdsk_qpsi_vs_flux_map_max_rel_error` | the file's `qpsi` against the q of its traced surfaces |
| `pest_jacobian_max_rel_error` | √g against q ψ_p′ R²/F, which holds in exact PEST coordinates |
| `radial_fit_residual_m`, `radial_degree` | radial fit of R and Z: the residual is the noise of the traced surfaces |
| `iota_fit_max_rel_error`, `iota_degree` | fit of ι(ρ) to the surface q values |

**`--method desc`.** This re-solves a fixed-boundary DESC equilibrium inside ψ_N = `--psin-b`, as `compute2D_0170.ipynb` does for analytic inputs.

- ι = 1/|q| and p are even power series (`--iota-order`, `--pressure-order`). `--M` sets the resolution and `--solver` the solver.
- The DESC equilibrium is saved as `<output>_desc.h5`.
- Checks are stored under `inputs/desc_checks`: axis position, \|B\| on axis, q, the g-file ψ_N on DESC surfaces, and the force error.

This method imposes the file's `qpsi`. For DIII-D 204232, `qpsi` is 0.64% off the flux map, and DESC's axis ends up 2–5 cm from EFIT's, depending on resolution.

**Pressure.** Both methods use the g-file's `pres` by default.

- `--pressure pprime` integrates the file's p′, the pressure its Grad–Shafranov solution actually used. In some files `pres` disagrees with it.
- `--pressure kinetic` uses the sum over the kinetic profiles.

It also prints the on-axis pressure from `pres`, from p′ and from the kinetic profiles. If they disagree, the g-file is inconsistent, or the profiles and the equilibrium describe different plasmas.

### Kinetic profiles

Profiles are named as in `NTP.mat`, with the same units:

- `ne Te ni Ti`;
- `Pi`, only for a slowing-down thermal species;
- `na Ta Pa` and `nb Tb Pb` for the alpha and beam slots.

Densities are in 1e19 m⁻³ and pressures in Pa. `T` is in keV for a Maxwellian species, and is the critical speed v_c in m/s for a slowing-down one.

The radial coordinate is one of:

- `psi_n` (g-file normalized poloidal flux);
- `rho_pol` = √ψ_N;
- `rho_tor` (toroidal flux, normalized to the g-file boundary);
- `rho` (ρ = √(ψ_t/ψ_t(ψ_N,b)); with `--method pycocos`, ψ_t comes from the flux-map q).

Points beyond ρ = 1 are dropped. For diverted plasmas, prefer `psi_n` or `rho_pol`: `rho_tor` depends on the g-file's q at the separatrix, which is large and poorly defined.

Accepted files:

- HDF5: one coordinate dataset and the profile datasets, at the root or in `/profiles`.
- `NTP.mat`: `rhoSample` is DESC ρ.
- A text table: the first line is a header, `psi_n ne Te ni Ti ...`, followed by columns separated by spaces or commas.

### prepare

- Reads `standard2D.mat` and `NTP.mat` and writes the equilibrium HDF5 file.
- `--types ION ALPHA BEAM` takes 1 (Maxwellian), 2 (slowing-down) or 3 (absent), the codes of `generateInput2D.m`. Thermal ions are required.
- `--fit NAME=WINDOW,ORDER` sets a profile fit, as `*Window` and `*Order` in `generateInput2D.m`. The default is order 12 over all samples.

### case

`case` fills these settings in a copy of the template:

- the equilibrium constants: B0, L0, VA0, RHO0, RHO1, PSITMAX, gridNx, gridNy, NFP, the three betas and IonMass;
- `inputDir` set to the case directory, and `outputDir` set to `<case>/output`;
- `MHDEquilibrium`.

`--set NAME=VALUE` changes any other constant or switch.

`case` stops if `if<Species>` or `<Species>Type` contradict the species in the equilibrium. In the file, the T block holds keV for a Maxwellian but v_c/VA0 for a slowing-down distribution. It also warns about grid sizes that don't divide the CUDA launch geometry in `src/cuGMEC_setup.h`.

### outputs

- Reads the run's own `final/cuGMEC_param.h`. `make` embeds the parameter file in the executable and the run writes it there.
- Files whose switch is off, or whose size does not match, are reported and skipped.
- `--restart` also stores `MHDContinue` and the `PICContinue` markers, which can be large.

### Other commands

- `cugmec export-bin eq.h5 -o file.bin` writes the binary only.
- `cugmec import-bin file.bin --param cuGMEC_param.h -o eq.h5` wraps an existing binary, for example one made by MATLAB.

## HDF5 layouts

**Equilibrium** (`format = "cuGMEC equilibrium"`):

| Path | Content |
|---|---|
| `grid/` | `x`, `rho`, `y`, `y_ghost`; attributes `gridNx`, `gridNy`, `equilNz`, `NFP`, `ghost` |
| `normalization/` | B0, L0, VA0, RHO0, RHO1, PSITMAX and the betas, with `unit` and `description` |
| `blocks/<name>` | the 159 blocks in cuGMEC's normalized form, shape (Nx, ny, equilNz), named as in `src/cuGMEC_model.h`. Attributes `description`, `normalization`, `index` |
| `inputs/` | species types, ion mass, fit settings, source files with SHA-256, and the profile samples |

**Output** (`format = "cuGMEC output"`). Every dataset has `dims`, `description`, `source` and, where known, `unit`.

| Path | Content |
|---|---|
| `param/` | every constant and switch of the run as attributes, plus the file text |
| `grid/`, `time/` | coordinates. Time is in L0/VA0 and includes `continueSteps`. `grid.attrs.diag_y_index` is the y plane of the 1D diagnostics |
| `equilibrium/` | the equilibrium the run read, in the layout above |
| `diagnostics/` | amplitude, RealMode, ImagMode, frequency, shearing, Epara, EparaES, zonal drives, `<Sp>Density`, `<Sp>Diffusivity` |
| `fields/<X>` | `total<X>.bin`: (time_output, y, x, z) |
| `final/<X>` | state at the last step: (y, x, z) |
| `initial/<Sp>/` | equilibrium density and pressure from the loaded markers |
| `phase_space/<Sp>/` | axes `E`, `Pphi`, `Lambda` (Pphi recomputed from the equilibrium, as in the code); `f0`, `jacobian`, `orbit`, `delta_f`, `power` |
| `pitch_space/<Sp>/` | axes `vpara`, `vperp`; `f0`, `jacobian`, `delta_f`, `power` |
| `restart/` | with `--restart`: MHD restart fields and the markers (x, y, z, vpara, w, p, mu, cell) |

## Differences from the MATLAB scripts

- The shifted-metric (SFT) and field-aligned (SFA) metrics come from one Jacobian-matrix transform (`field_aligned_metric`). The results are identical to rounding error.
- J∥ is differentiated with a 5th-order spline in ρ and an FFT in θ instead of the B-spline MEX. J∥/B differs by about 2e-7.
- Output times include `continueSteps`; `visualizeMHD.m` and `visualizePIC.m` start continued runs at t = 0.

## Tests

```bash
cd scripts/python
python -m pytest tests                                    # or, without pytest:
PYTHONPATH=. python tests/test_example.py && PYTHONPATH=. python tests/test_outputs.py
```

- `test_example.py` rebuilds `example/ITPA_TAE_128_32.bin` and compares all 159 blocks with the MATLAB output. The alpha density is skipped because the fit settings used for the example are not recorded. It also tests the HDF5 round trip and `cugmec case`.
- `test_outputs.py` converts a synthetic run directory whose files hold their own flat index, and checks the decoded values.
- `test_pest.py` runs the pyCOCOS path on a synthetic g-file with concentric circular surfaces, where q, the PEST angle, \|B\| and J∥ are known analytically. It also runs `from-geqdsk` end to end.
- `test_geqdsk.py` tests the g-file reader, the flux maps and the profile formats on a synthetic g-file. With DESC installed, it also checks that `desc_eval.standard2d` reproduces the notebook's `outputStandard2D` exactly.

## Not covered yet

- The 3D (stellarator) preprocessing: `DESC2PEST2SFA3D.m` and `writeEquilibrium3D.m`.
- The phase-space mapping input: `generatePhaseSpaceMapping2D.m`.
- The region beyond ψ_N = `--psin-b`: both methods stop short of the separatrix in diverted plasmas.
- Output conversion has not yet been run on a real cuGMEC run.
