# Performance

These single-GPU timings are those published in the cuGMEC README by its authors. They supersede the CPC paper
{cite:p}`liu2026cugmec`, since both the MHD and the PIC parts have been optimized since.

**Benchmark.** The ITPA TAE case ([](examples/itpa.md)) on a $256\times64\times16$ grid with
$\Delta t = 0.025\,R_0/v_A$. The MHD part runs in double precision. The PIC part uses 400 keV fast ions, with a largest
speed of about $1.2\,v_A$, in double or single precision; the deposition is done in double precision in both cases.
GYRO is the number of gyro-average points and P/G the average number of particles per grid point. All GPUs use the same
`gridDim` and `blockDim`, so the timings may differ from each GPU's optimum.

The run time of a simulation is about (MHD time per step + PIC time per step) × number of steps, and other sizes and GPU
counts follow by simple scaling.

## Single-GPU timings

### NVIDIA GeForce RTX 4090 D

MHD: 14.6 ms per step. PIC per step:

| P/G | double, GYRO 0 | 4 | 8 | 16 | float, GYRO 0 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|
| 32 | 30.5 ms | 102 ms | 165 ms | 296 ms | 2.82 ms | 8.96 ms | 15.0 ms | 28.5 ms |
| 64 | 60.1 ms | 203 ms | 329 ms | 584 ms | 4.27 ms | 15.5 ms | 26.1 ms | 49.7 ms |
| 128 | 119 ms | 403 ms | 656 ms | 1.16 s | 7.50 ms | 28.5 ms | 48.5 ms | 93.1 ms |
| 256 | 237 ms | 806 ms | 1.33 s | 2.31 s | 14.0 ms | 54.9 ms | 94.0 ms | 178 ms |

### NVIDIA A800-SXM4-80GB

MHD: 13.8 ms per step. PIC per step:

| P/G | double, GYRO 0 | 4 | 8 | 16 | float, GYRO 0 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|
| 32 | 7.26 ms | 26.3 ms | 44.4 ms | 78.4 ms | 3.72 ms | 13.9 ms | 23.1 ms | 42.0 ms |
| 64 | 13.5 ms | 48.9 ms | 83.8 ms | 148 ms | 6.65 ms | 25.4 ms | 41.7 ms | 75.8 ms |
| 128 | 26.2 ms | 94.2 ms | 162 ms | 288 ms | 12.5 ms | 48.6 ms | 79.8 ms | 143 ms |
| 256 | 51.2 ms | 184 ms | 319 ms | 566 ms | 24.4 ms | 95.7 ms | 161 ms | 281 ms |

### NVIDIA RTX PRO 6000 Blackwell Server Edition

MHD: 9.76 ms per step. PIC per step:

| P/G | double, GYRO 0 | 4 | 8 | 16 | float, GYRO 0 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|
| 32 | 22.4 ms | 76.5 ms | 125 ms | 222 ms | 1.68 ms | 6.24 ms | 9.71 ms | 25.4 ms |
| 64 | 41.0 ms | 140 ms | 229 ms | 407 ms | 2.65 ms | 10.4 ms | 17.0 ms | 43.6 ms |
| 128 | 81.6 ms | 280 ms | 457 ms | 809 ms | 4.09 ms | 18.7 ms | 31.0 ms | 66.3 ms |
| 256 | 162 ms | 556 ms | 906 ms | 1.61 s | 7.90 ms | 36.6 ms | 60.3 ms | 127 ms |

### NVIDIA B200

MHD: 9.89 ms per step. PIC per step:

| P/G | double, GYRO 0 | 4 | 8 | 16 | float, GYRO 0 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|
| 32 | 0.906 ms | 6.60 ms | 15.5 ms | 32.6 ms | 0.805 ms | 3.27 ms | 7.80 ms | 17.0 ms |
| 64 | 2.03 ms | 16.9 ms | 33.8 ms | 66.6 ms | 1.01 ms | 7.08 ms | 16.3 ms | 33.0 ms |
| 128 | 7.20 ms | 37.5 ms | 70.4 ms | 135 ms | 3.33 ms | 18.2 ms | 39.2 ms | 73.1 ms |
| 256 | 18.1 ms | 81.2 ms | 151 ms | 294 ms | 5.82 ms | 40.5 ms | 89.2 ms | 162 ms |
## Scaling to a production case

An ITER steady-state full-torus case up to toroidal mode number 36 uses a $384\times32\times576$ grid (about 7.1 million
points), P/G = 240 (about 5.1 billion particles in three ion species), GYRO = 4, $\Delta t = 0.025$ and 20 000 steps,
with double-precision MHD and single-precision PIC. Linear scaling from the tables predicts 5.31 h on 8 A800 GPUs
(measured: 4.97 h, 6 % less) and 2.32 h on 8 B200 GPUs (measured: 2.23 h, 4 % less).

## Local ITPA run

The ITPA run of [](examples/itpa.md) ($128\times32\times16$, $2^{22}$ markers in the alpha slot, GYRO = 4, single-precision
PIC, linear) took 312 s for 6000 steps, 52 ms per step, on an RTX 3050 Laptop GPU (4 GB, 35 W), using 1.4 GB of device
memory.
