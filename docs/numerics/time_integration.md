# Time integration

## Fluid step

The fluid fields $(\varpi, A, \delta n_e, \delta T_e, \delta p)$ are advanced with classical fourth-order Runge–Kutta,
using four buffers (begin, two intermediate, end):

| Stage | Input | Writes |
|---|---|---|
| 1 | $y_n$ | $y^{(2)} = y_n + \tfrac{\Delta t}{2}k_1$; $y_\mathrm{end} = y_n + \tfrac{\Delta t}{6}k_1$ |
| 2 | $y^{(2)}$ | $y^{(3)} = y_n + \tfrac{\Delta t}{2}k_2$; $y_\mathrm{end} \mathrel{+}= \tfrac{\Delta t}{3}k_2$ |
| 3 | $y^{(3)}$ | $y^{(4)} = y_n + \Delta t\,k_3$; $y_\mathrm{end} \mathrel{+}= \tfrac{\Delta t}{3}k_3$ |
| 4 | $y^{(4)}$ | $y_{n+1} = y_\mathrm{end} + \tfrac{\Delta t}{6}k_4$ |

After each of stages 1–3 the derived fields are recomputed: the Poisson solve $\varpi\to\Phi$, quasi-neutrality if
enabled, $\tilde J(A)$, $\delta p_e$ and the FLR correction of $\Phi$, then a halo exchange of all fields.

After stage 4, a "stage 5" closes the step:

1. Poisson solve;
2. operator-split dissipation, then the mode filters;
3. $\varpi$ recomputed from the processed $\Phi$, then quasi-neutrality, $\tilde J$, $\delta p_e$, FLR, and a halo
   exchange;
4. optional NaN check, diagnostics and output.

The kinetic pressures $\delta p_s$ (and densities $\delta n_s$) are held fixed during the fluid stages.

## Coupling with the particles

The main loop alternates the two systems:

1. `ratioDt` fluid RK4 steps, with $\delta p_s$ frozen at their last deposit;
2. preparation of the fields for the particles (field-aligned conversion, all-gather, derivatives, $\partial_t A$);
3. one particle RK4 step of $\Delta t_\mathrm{PIC}$ = `ratioDt`·`dt` for each species, with push and deposit in one
   kernel, in the fields frozen at the end of the fluid steps;
4. deposit reduction across GPUs, ghost folding, conversion to the shifted representation, dissipation and filters on
   $\delta p_s$, halo exchange;
5. with `ifQNeutrality`, update of $\delta n_e$, $\tilde J$ and $\delta p_e$.

The markers are sorted every `sortSteps` particle steps.

Each subsystem is fourth order internally, but the coupling is explicit and lagged, hence first order in the coupling
interval. GMEC used the same step for fluid and particles, about $0.01\,\tau_A$ {cite:p}`liu2024gmec2`; the cuGMEC
examples use $\Delta t = 0.02$–$0.05\,t_0$ with `ratioDt = 1`.

## Precision

| Quantity | Precision |
|---|---|
| fluid fields, coefficient arrays, solves | `mhdReal` (double in the examples); coefficients are built in double on the host |
| markers and the field tables they read | `picReal` (float in the examples) |
| deposit buffers and their atomic additions | `mhdReal` |

`mhdReal` must be at least as precise as `picReal`.
