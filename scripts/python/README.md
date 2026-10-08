# cugmec: Python tools for cuGMEC

The `cugmec` package prepares cuGMEC inputs and converts its outputs. HDF5 is the format you work with. The binary files cuGMEC reads and writes are generated from HDF5 or converted into it. It replaces the MATLAB stage of the 2D (tokamak) preprocessing: `generateInput2D.m`, i.e. `writeProfile1D`, `DESC2PEST2SFA2D`, `writeNormalization2D` and `writeEquilibrium2D`.

```bash
pip install -e scripts/python        # needs numpy, scipy, h5py; adds the `cugmec` command
```

## Workflow

Each case gets its own directory, its own `cuGMEC_param.h` and its own executable.

```bash
# 1. DESC equilibrium -> standard2D.mat with scripts/equilibrium/output2D_0170.ipynb, as before
# 2. equilibrium + profiles -> equilibrium HDF5
cugmec prepare standard2D.mat NTP.mat -o itpa_eq.h5 --ion-mass 1.0 --types 1 1 3
# 3. case directory: cuGMEC_param.h from a template, plus the binary the code reads
cugmec case itpa_eq.h5 runs/itpa --template example/cuGMEC_param.h --set dt=0.05 --set ifOutputPhi=true
# 4. build and run (from the repository root)
make CASE=runs/itpa
mpirun -n 1 runs/itpa/cuGMEC
# 5. outputs -> HDF5
cugmec outputs runs/itpa/output -o runs/itpa/itpa.h5
```

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

## Not covered yet

- The 3D (stellarator) preprocessing: `DESC2PEST2SFA3D.m` and `writeEquilibrium3D.m`.
- The phase-space mapping input: `generatePhaseSpaceMapping2D.m`.
- Output conversion has not yet been run on a real cuGMEC run.
