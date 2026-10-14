# ITPA $n = 6$ TAE benchmark

The ITPA energetic-particle group's TAE case {cite:p}`konies2018` is the standard linear benchmark of hybrid and
gyrokinetic codes: an $n = 6$ toroidal Alfvén eigenmode driven by Maxwellian fast deuterons in a large, circular,
low-$\beta$ tokamak. cuGMEC ships its input as the example (`example/ITPA_TAE_128_32.bin`, with
`example/cuGMEC_param.h`), and GMEC's results for the whole temperature scan are in {cite:t}`liu2024gmec2`.

## The benchmark case

| Quantity | Value |
|---|---|
| geometry | $R_0 = 10$ m, $a = 1$ m, circular, $B_0 = 3$ T |
| safety factor | $q(r) = 1.71 + 0.16\,(r/a)^2$ |
| bulk plasma | hydrogen, flat $n_i = n_e = 2\times10^{19}$ m⁻³, $T_e = T_i = 1$ keV |
| bulk pressure (hybrid codes) | $p(s) = 7.17\times10^3 - 6.811\times10^3 s - 3.585\times10^2 s^2$ Pa, $s = \psi_\mathrm{tor}/\psi_\mathrm{tor}(a)$ |
| fast ions | deuterons, isotropic Maxwellian, $T_h = 400$ keV here |
| fast-ion density | $n_h(s) = n_0 c_3\exp\left[-\tfrac{c_2}{c_1}\tanh\tfrac{\sqrt s - c_0}{c_2}\right]$, $n_0 = 1.44131\times10^{17}$ m⁻³, $c_0 = 0.49123$, $c_1 = 0.298228$, $c_2 = 0.198739$, $c_3 = 0.521298$ |
| mode | $n = 6$, dominated by $m = 10$ and $11$, peaked near $\sqrt s \approx 0.5$ where $q = 10.5/6$ |

The ideal-MHD frequency of the TAE is $4.01\times10^5$ rad/s with CAS3D and $4.13\times10^5$ rad/s with GYGLES. The
growth rates of the participating codes agree within about ±15 % without finite-Larmor-radius effects and ±20 % with
them; FLR effects lower the growth rate {cite:p}`konies2018`.

## Set-up

The run uses the example input and parameter file, with the model made linear and the thermal ions treated as a fluid
only:

| Setting | Value |
|---|---|
| equilibrium | `ITPA_TAE_128_32.bin`, a DESC equilibrium with the profiles above; $\rho\in[0.1, 0.9]$ ($\rho = \sqrt{\psi_t/\psi_{t,a}}$), $\psi_{t,a}/2\pi = 1.5$ Wb/rad |
| normalization | $L_0 = 10.013$ m, $B_0 = 2.993$ T, $v_{A0} = 1.460\times10^7$ m/s, $t_0 = 0.686$ µs |
| grid | $128\times32\times16$ ($x$, $y$, $z$), `tubes = 6` (one sixth of the torus), so the retained mode is $n = 6$ |
| fluid model | linear (`ifNonlinearMHD = false`), ion diamagnetic drift and parallel electric field on (`ifDiaDrift`, `ifEparallel`), thermal-ion $\beta = 9.0\times10^{-4}$ (at $\rho_0$); no dissipation |
| fast ions | alpha slot: Maxwellian deuterons ($A = 2$, $Z = 1$), $\beta_h = 2.57\times10^{-3}$ at $\rho_0 = 0.1$, $v\in[0.065, 1.3]\,v_{A0}$, `spaceReal` and `velocityReal` loading, 64 markers per cell ($2^{22}$ markers, of which $2^{18}$ independent), linear $\delta f$ |
| FLR | four-point gyro-average for the particles (`ifFLRPIC`, `gyroNums = 4`); none in the Poisson equation (`ifFLRMHD = false`) |
| thermal ions | not kinetic (`ifIon = false`) |
| time step | $\Delta t = 0.05\,t_0$, 6000 steps to $t = 300\,t_0$ (206 µs) |
| initial perturbation | Gaussian in $x$ at index 64 ($\rho\approx0.5$), width 0.05, $\Phi$ amplitude $10^{-9}$ |
| precision | double for the MHD, single for the particles |

Relative to `example/cuGMEC_param.h`, only `ifNonlinearMHD`, `ifNonlinearPIC` and `ifIon` were set to `falseType`, and
`diagSteps` to 10. On an RTX 3050 Laptop GPU (4 GB) the run took 312 s (52 ms per step) and 1.4 GB of device memory.

```bash
cp example/cuGMEC_param.h runs/itpa/cuGMEC_param.h   # then edit inputDir, outputDir and the switches above
make CASE=runs/itpa && mpirun -n 1 runs/itpa/cuGMEC
cugmec outputs runs/itpa/output -o runs/itpa/ITPA_ref.h5
python docs/scripts/make_figures.py runs/itpa/ITPA_ref.h5
```

## Results

```{figure} ../figures/itpa/amplitude.png
:name: fig-cugmec-itpa-amplitude
:width: 85%

Largest $|\Phi_{n=6}|$ over the radius on the diagnostic plane, against time, with the exponential fit over
$t \in [180, 300]\,t_0$.
```

The beating of the initial transient dies out by $t \approx 150\,t_0$, and the mode then grows exponentially
({numref}`fig-cugmec-itpa-amplitude`). The fit over the last 40 % of the run gives

$$
\gamma = 0.0176\,\omega_{A0} = 2.57\times10^4\ \mathrm{s^{-1}},\qquad
\omega = 0.282\,\omega_{A0} = 4.11\times10^5\ \mathrm{rad/s}\ (65.4\ \mathrm{kHz}).
$$

The growth rates over the two halves of the window agree to $5\times10^{-4}$ (relative), so the mode is a clean
eigenmode. Its frequency lies between the ideal-MHD values of CAS3D and GYGLES.

```{figure} ../figures/itpa/radial.png
:width: 85%

Radial profile of $|\Phi_6|$ on the diagnostic plane ($\theta\approx0$, outboard midplane) at three times, each
normalized to its maximum.
```

The mode peaks at $\rho = 0.50$, where $q = 1.75 = 10.5/6$, as expected for the $m = 10$–11 TAE, and its radial
structure no longer changes after the transient.

```{figure} ../figures/itpa/harmonics.png
:width: 85%

Poloidal harmonics of $\Phi_6$ (PEST angle) at the end of the run, normalized to the largest.
```

The poloidal spectrum is that of a textbook TAE: two harmonics of almost equal amplitude, $m = 10$ (peak 1.00) and
$m = 11$ (0.95), centred just inside and just outside the gap at $\rho = 0.5$; the sidebands $m = 9$ and $12$ are at
3 %. The harmonics are computed from the final $\Phi$ by FFTs in $z$ and in $y$: the stored field is on the PEST angle,
with 32 points that resolve $|m| \le 16$.

```{figure} ../figures/itpa/mode.png
:width: 65%

$\Phi_6$ in the poloidal plane $\varphi = 0$ at the end of the run, rebuilt from the poloidal harmonics. The white disc
and the outer circle are the boundaries of the simulation domain, $\rho = 0.1$ and $0.9$.
```

The mode balloons on the low-field side, where the $m = 10$ and $11$ harmonics add in phase.

```{figure} ../figures/itpa/loading.png
:width: 80%

Fast-ion density of the loaded markers, averaged over $y$ and $z$, against the ITPA formula.
```

The loaded fast-ion density follows the ITPA profile, with marker noise of a few per cent that reaches 10 % near the
inner boundary, where the cells are smallest. The markers are replicated over the $z$ planes, so $2^{18}$ of the
$2^{22}$ markers are independent.

## Comparison with cuMEGA

cuMEGA solves MEGA's compressible single-fluid MHD in cylindrical coordinates, with guiding-centre fast ions. Its run
of the same case (TokaMaker g-file of the ITPA equilibrium, $128\times128\times16$ grid with $N_\mathrm{per} = 6$,
524 288 markers, exponential time integration with $\Delta t = 0.05$, dissipation $10^{-8}$) predates cuMEGA's
finite-Larmor-radius option and was made without FLR; it is described in cuMEGA's own documentation.

| | $\omega/\omega_{A0}$ | $\omega$ [$10^5$ rad/s] | $f$ [kHz] | $\gamma/\omega_{A0}$ | $\gamma$ [$10^4$ s⁻¹] |
|---|---|---|---|---|---|
| cuGMEC (gyrokinetic, FLR) | 0.282 | 4.11 | 65.4 | 0.0176 | 2.57 |
| cuMEGA (MHD, no FLR) | 0.276 | 4.02 | 64.0 | 0.022 (0.018–0.026) | 3.2 (2.7–3.9) |
| CAS3D / GYGLES, ideal MHD | | 4.01 / 4.13 | | | |

The frequencies agree within 2 %. cuMEGA's growth rate is about 25 % higher, as expected without FLR, and less
precise, because its run ends while the TAE is still a small part of the signal.
