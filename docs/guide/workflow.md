# Workflow

A cuGMEC simulation goes through five stages. Each case is compiled with its own parameter file, because every run
parameter is a compile-time constant.

| Stage | MATLAB / notebook route | Python route (tokamaks) |
|---|---|---|
| 1. equilibrium | `scripts/equilibrium/compute2D_0170.ipynb` (DESC), exported with `output2D_0170.ipynb` | `cugmec from-geqdsk` (pyCOCOS, straight from a g-file) or `cugmec from-desc` |
| 2. input file | `scripts/preprocess/generateInput2D.m` (`generateInput3D.m` for stellarators); `generatePhaseSpaceMapping2D.m` for the phase-space diagnostics | the same command writes the equilibrium HDF5 file and the binary |
| 3. parameters | edit `src/cuGMEC_param.h` ([](../en-US/parameters.md)) | `cugmec case` / `--case` fills a copy of a template in the case directory |
| 4. build and run | configure the `Makefile` ([](../en-US/environment.md)), `make`, `mpirun` | `make CASE=<dir>`, `mpirun -n <N> <dir>/cuGMEC` |
| 5. post-processing | `scripts/postprocess/visualizeMHD.m`, `visualizePIC.m` | `cugmec outputs <dir>/output -o run.h5` |

## Python route

```bash
mamba activate cugmec
# 1-3: gEQDSK + kinetic profiles -> equilibrium HDF5, binary and case directory
cugmec from-geqdsk g.eqdsk profiles.txt -o eq.h5 --ion-mass 2 --types 1 2 3 \
       --case runs/x --template example/cuGMEC_param.h --set AlphaType=Slowing0
# 4: build and run (from the repository root)
make CASE=runs/x
mpirun -n 1 runs/x/cuGMEC
# 5: outputs -> HDF5
cugmec outputs runs/x/output -o runs/x/x.h5
```

See [](python.md) for the commands, the checks they make, and the HDF5 layouts.

## Building

`make CASE=<dir>` compiles `src/cuGMEC_main.cu` with `<dir>/cuGMEC_param.h` into `<dir>/cuGMEC`. Without `CASE`, the
build uses `src/cuGMEC_param.h`. The dependencies are CUDA 12 (with cuFFT and cuSPARSE), NCCL, cuDSS 0.8 and MPI
([](../en-US/environment.md)). On a Cray system the Makefile takes MPI and the CUDA math libraries from the cray-mpich
and cudatoolkit modules. On Perlmutter, the `cugmec` module in `/global/common/software/m5040/modules` sets up the
environment; see its `WORKFLOW.md`.

## Running and restarting

```bash
mpirun -n <hostNums> <dir>/cuGMEC
```

The number of MPI processes must equal `hostNums`, and each process drives `devNums` GPUs. The run reads
`inputDir/MHDEquilibrium` and writes to `outputDir`:

- `outputDir/final/`: diagnostics (time traces), the final fields, the restart files and a copy of the parameter file;
- `outputDir/initial/`: equilibrium density and pressure of each kinetic species, computed from the loaded markers.

To continue a run, copy the `MHDContinue` and `PICContinue` files of `outputDir/final` into the new `inputDir`, set
`ifContinue = trueType` and `continueSteps` to the steps already done, and rebuild.

## Contact

Using cuGMEC generally requires nontrivial pre- and post-processing. The developers recommend contacting them before
starting: [s.y.liu@zju.edu.cn](mailto:s.y.liu@zju.edu.cn), [29.sy.liu@gmail.com](mailto:29.sy.liu@gmail.com).
