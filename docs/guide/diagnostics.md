# Diagnostics and outputs

The run writes binary files to `outputDir/final` and `outputDir/initial`. `cugmec outputs` converts them into one HDF5
file with named, documented datasets ([](python.md)); the paths below are those of the HDF5 file. Time is in
$t_0 = L_0/v_{A0}$ (`time.attrs.seconds_per_unit` gives $t_0$ in seconds) and includes `continueSteps`.

## Time traces (every `diagSteps` steps)

The one-dimensional diagnostics are taken on one $y$ plane, `grid.attrs.diag_y_index`: plane `gridNy`/2 (just above
the outboard midplane $\theta = 0$) with one GPU, otherwise the first plane of the middle GPU.

The code computes no energies (kinetic, magnetic or particle); they have to be computed in post-processing from the
field outputs.

| Dataset | Switch | Content |
|---|---|---|
| `diagnostics/amplitude` | `ifDiagAmplitude` | $\lvert\mathrm{FFT}_z\Phi\rvert$ per retained toroidal mode, against $x$ (normalized so that a single mode $A\cos(n\varphi)$ gives $A$) |
| `diagnostics/RealMode`, `ImagMode` | `ifDiagAmplitude` | real and imaginary parts of the unnormalized $\mathrm{FFT}_z\Phi$, for the phase and the frequency |
| `diagnostics/frequency` | `ifDiagFrequency` | $\Phi$ at $z$ index 0 against $x$: a time signal for spectral analysis |
| `diagnostics/Epara`, `EparaES` | `ifDiagEparallel` | the parallel electric field, from Ohm's law, and its electrostatic part $\nabla_\parallel\Phi$, at $z$ index 0; their difference is the inductive part $\partial_t A_\parallel$ (signs as in the dataset descriptions) |
| `diagnostics/<Sp>Density` | `ifDiagDensity` | radial profile of $\delta n$ of each kinetic species |
| `diagnostics/<Sp>Diffusivity` | `ifDiagDiffusivity` | radial particle diffusivity of each kinetic species, per toroidal mode, $D = -\langle\Gamma\cdot\nabla x\rangle/(\langle\lvert\nabla x\rvert^2\rangle\,\partial_x n)$ in m²/s |
| `diagnostics/MaxwellDrive`, `ReynoldsDrive`, `ZonalDrive` | `ifDiagZFDrive` | zonal-flow drive by the Maxwell and Reynolds stresses, and the change of the zonal vorticity, per step |
| `diagnostics/shearing` | `ifDiagShearing` | Hahm–Burrell $E\times B$ shearing rate $\langle(RB_p)^2/B\rangle\,\partial^2\Phi_{00}/\partial\psi_p^2$ from the $(n, m) = (0, 0)$ part of $\Phi$, in s⁻¹ |

With `ifCheckNAN`, the run checks the fields for NaN and stops, writing its output, as soon as one appears.

## Fields

| Dataset | Switch | Content |
|---|---|---|
| `fields/<X>` | `ifOutput<X>` | $X$ = `Phi`, `A`, `dNe`, `dTe`, `dP`, `dPi`, `dPa`, `dPb` every `outputSteps`, shape (time, $y$, $x$, $z$) |
| `final/<X>` | always | every field at the last step, shape ($y$, $x$, $z$) |

The fields are stored on the grid of [](../physics/coordinates.md): $y$ is the PEST poloidal angle at the cell centres of
each $y$ plane, $x$ the radial coordinate, and $z$ the toroidal angle over $2\pi/$`tubes`. The ITPA example shows how to
take poloidal harmonics and plot a poloidal plane from them ([](../examples/itpa.md)).

## Kinetic species

| Dataset | Switch | Content |
|---|---|---|
| `initial/<Sp>/density`, `pressure` | always (first run) | equilibrium density (m⁻³) and pressure (Pa) of the loaded markers, averaged over $y$ and $z$ |
| `phase_space/<Sp>/jacobian`, `f0`, `orbit` | `ifOutputPhaseSpace*` | on a grid in energy $\mathcal E$, canonical toroidal momentum $P_\varphi$ and $\Lambda = \mu B_0/\mathcal E$ (`gridE`, `gridPphi`, `gridLambda`): markers per node, $f_0$, and the orbit (frequencies and type) of each node |
| `phase_space/<Sp>/delta_f`, `power` | `ifOutputPhaseSpaceDeltaF`, `Power` | $\delta f$, and the wave–particle power per toroidal mode (W; positive when the particles give energy to the mode), every `outputSteps` |
| `pitch_space/<Sp>/...` | `ifOutputPitchSpace*` | the same on a $(v_\parallel, v_\perp)$ grid (`gridVpara`, `gridVperp`) |

The phase-space diagnostics need a mapping file per species (`<Sp>PhaseSpaceMapping`), made by
`scripts/preprocess/generatePhaseSpaceMapping2D.m` with the same grid sizes.
The particle diagnostics (density, diffusivity, phase and pitch space) are implemented for tokamaks only.

## Visualization

`scripts/postprocess/visualizeMHD.m` and `visualizePIC.m` plot every diagnostic above: amplitude and frequency
against time, mode structures on flux surfaces and in the poloidal plane, poloidal spectra, phase-space distributions
and power transfer. They read the binary files directly.
