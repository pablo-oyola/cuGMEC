# Parallelization

## Process layout

A run uses `hostNums` MPI processes, each driving `devNums` GPUs, with one NCCL communicator over all
`hostNums`·`devNums` GPUs. `gridNy` must be divisible by `hostNums`·`devNums`.

## Fluid part: domain decomposition in $y$

The $y$ planes are split into contiguous slabs of `gridNy`/(`hostNums`·`devNums`) planes, each with two ghost planes on
each side. Halo exchanges are periodic ring `ncclSend`/`ncclRecv` operations (with a layered variant when a slab has
fewer than two planes). Everything else in the fluid step is local to a GPU: the Poisson and dissipation solves are per
plane, and the toroidal FFTs are along $z$. Only the poloidal filter `selectNM` all-gathers its envelopes.

## Particle part: particle decomposition

Each GPU holds `gridNx`·`gridNy`·`gridNz`·`ppcNums`/(`hostNums`·`devNums`) markers per species, which range over the
whole domain. Before each particle step the field tables are all-gathered over $y$, so every GPU has the full fields;
after the push, the deposits are summed with `ncclAllReduce`. This keeps the particle part free of marker migration, at
the cost of a full-domain field table per GPU.

Each thread pushes 32 markers with coalesced indexing, and markers are periodically sorted by cell for locality of the
gathers and of the atomic deposits.

## Diagnostics

The one-dimensional time traces are sampled on one $y$ plane of one GPU: plane `gridNy`/2 with a single GPU, otherwise
the first local plane of the middle device.

## Build per case

Every run parameter is a compile-time constant (`src/cuGMEC_param.h`, or `<case>/cuGMEC_param.h` with
`make CASE=<case>`), which lets the compiler specialize the kernels: switches are types (`trueType`/`falseType`) and
the grid sizes are constants. The text of the parameter file is compiled into the executable and written to
`outputDir/final/cuGMEC_param.h` by every run.
