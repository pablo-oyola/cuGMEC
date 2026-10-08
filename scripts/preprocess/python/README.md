# Python preprocessing (2D)

`cugmec_eq.py` replaces the MATLAB stage of the tokamak preprocessing (`generateInput2D.m`, i.e. `writeProfile1D`, `DESC2PEST2SFA2D`, `writeNormalization2D`, `writeEquilibrium2D`). It reads the same inputs and writes the same equilibrium file.

```
DESC equilibrium ── scripts/equilibrium/output2D_0170.ipynb ──> standard2D.mat ─┐
                                                       NTP.mat (profiles) ──────┴─ cugmec_eq.py ──> MHDEquilibrium_Nx_Ny.bin
                                                                                                └─> constants for cuGMEC_param.h
```

Requires `numpy` and `scipy`. MATLAB and the `bspline` MEX are not needed.

## Usage

```bash
python cugmec_eq.py standard2D.mat NTP.mat MHDEquilibrium_128_32.bin --ion-mass 1.0 --types 1 1 3
```

- `--types ION ALPHA BEAM`: 1 Maxwellian, 2 slowing-down, 3 absent (the codes of `generateInput2D.m`). Thermal ions are required.
- `--fit NAME=WINDOW,ORDER`: profile fit settings, as `*Window` and `*Order` in `generateInput2D.m`. The default is order 12 over all samples. Example: `--fit nb=22,6 --fit Pb=22,6`.
- `--param-in src/cuGMEC_param.h --param-out case/cuGMEC_param.h`: write a copy of `cuGMEC_param.h` with B0, L0, VA0, RHO0, RHO1, PSITMAX, gridNx, gridNy, NFP and the three betas filled in. Without these options the declarations are printed instead.

The same steps are available from Python through `build_2d()`, `write_bin()`, `read_bin()` and `patch_param_h()`.

## Settings that must agree with `cuGMEC_param.h`

cuGMEC does not check any of these when it loads the file:

- `--types` must match `if<Species>` and `<Species>Type`. The `T` block holds keV for a Maxwellian and the critical speed over VA0 for a slowing-down distribution.
- `--ion-mass` must equal `IonMass`, because it enters VA0 and the `Va` and `Rho` blocks.
- `gridNx` and `gridNy` must match the file. The file has no header, and its size is `gridNx * (49 * (gridNy + 4) + 110 * gridNy) * 8` bytes. `read_bin()` refuses a mismatch.

## Differences from the MATLAB scripts

- The shifted-metric (SFT) and field-aligned (SFA) metrics come from one Jacobian-matrix transform in `field_aligned_metric()`, instead of the expanded symbolic expressions. The results are identical to rounding error.
- J∥ is differentiated with a 5th-order spline in ρ and an FFT in θ, instead of the B-spline MEX. J∥/B differs by about 2e-7.
- The DESC-vs-transform comparison plots are not reproduced.

## Verification

```bash
python verify_example.py
```

This rebuilds `example/ITPA_TAE_128_32.bin` from `example/standard2D.mat` and `example/NTP.mat` and compares all 159 blocks with the MATLAB output. It also compares the constants with `example/normalization2D.mat`. The alpha density is skipped because the fit settings used for the example are not recorded.

## Not covered yet

- The 3D (stellarator) path: `DESC2PEST2SFA3D.m` and `writeEquilibrium3D.m`.
- The phase-space mapping files: `generatePhaseSpaceMapping2D.m`.
