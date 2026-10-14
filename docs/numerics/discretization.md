# Spatial discretization

## Finite differences

All derivatives use five-point, fourth-order stencils:

| Derivative | Stencil |
|---|---|
| $\partial_x$, interior | $(f_{-2} - 8f_{-1} + 8f_{+1} - f_{+2})/(12\Delta x)$ |
| $\partial_x$ at $i = 0$, $N_x - 1$ | one-sided $(-25, 48, -36, 16, -3)/(12\Delta x)$ |
| $\partial_x$ at $i = 1$, $N_x - 2$ | biased $(-3, -10, 18, -6, 1)/(12\Delta x)$ |
| $\partial_x^2$, interior | $(-1, 16, -30, 16, -1)/(12\Delta x^2)$ |
| $\partial_x^2$ near the walls | $(35, -104, 114, -56, 11)$ and $(11, -20, 6, 4, -1)$, over $12\Delta x^2$ |
| $\partial_z$, $\partial_z^2$ | periodic central stencils |
| $\partial_x\partial_z$ | $\partial_z$ of $\partial_x$ |
| $\partial_y$, $\partial_y^2$ | along the field line (below) |

When `gridNz` is 8, 16 or 32, the $z$ stencils and the $z$ interpolations exchange values between the threads of a warp
(shuffles) instead of reading global memory.

## The parallel direction

The derivative along $y$ follows the field line ([](../physics/coordinates.md)): the values on the planes $j\pm1$,
$j\pm2$ are interpolated in $z$ (four-point Lagrange) to the points
$\varphi + q(x)[\theta_{j\pm p} - \theta_j]$, and the central stencil is applied to them. $\mathbf b\cdot\nabla$ appears
only in this form, $(B^y/B)\,\partial_y$, acting on $\tilde J$ (vorticity and continuity), on $\Phi$ and $\delta n_e$
(Ohm's law), and on $A$ and the equilibrium through the $\delta\mathbf B$ terms. Perpendicular operators never
differentiate a perturbation in $y$.

## How the right-hand sides are evaluated

The fluid equations are not coded operator by operator. At start-up the code builds, for every grid point, "compressed
coefficient" arrays: the numerical prefactors of each derivative of each field in each equation, from the metric, the
field and the profiles (`HybridModel::compressCoefficient`, generated symbolically). The kernels contract these arrays
with the finite-difference derivatives of the fields. In a tokamak the arrays depend on $(x, y)$; in a stellarator on
$(x, y, z)$. The linear and nonlinear terms have separate arrays and kernels; the nonlinear kernel adds to the result of
the linear one.

## Radial boundaries

The right-hand-side kernels write only the interior points $0 < i < N_x - 1$. All fields therefore keep their initial
value of zero on both walls (homogeneous Dirichlet conditions), $\Phi$ is fixed to zero by the identity rows of the
Poisson matrix, and the deposit leaves the wall nodes empty. There is no axis: $\rho_0 > 0$.

## Mode filters

Applied after every MHD step to $\Phi$, $A$, $\delta n_e$, $\delta T_e$ and $\delta p$, and after every particle step to
the deposited $\delta p_s$ (and $\delta n_s$), each switched per field (`ifFilterN_<F>` etc.):

1. **Toroidal band-pass.** A real FFT along $z$ keeps the wedge harmonics $\ell\in[$`leftN`, `rightN`$]$, i.e. the
   toroidal mode numbers $n = \mathrm{tubes}\cdot\ell$. Because $z = \varphi$ in the shifted representation, this is an
   exact toroidal filter at fixed $(x, \theta)$.
2. **`removeN`.** Removes the listed harmonics $\ell$.
3. **Poloidal filter `selectNM`** $\{\ell, m_L, m_R\}$. For the harmonic $\ell$:
   - its complex envelope $E(x, y)$ is extracted and subtracted from the field;
   - it is rotated to the field-aligned frame, $E_A = E\,e^{-inq\theta}$, gathered over $y$ with twist-shifted ghosts,
     and interpolated onto a $y$ grid refined `refinedTimes` times;
   - it is rotated back, Fourier-transformed in $\theta$, and only $|m|\in[m_L, m_R]$ is kept;
   - the result is sampled at the original $y_j$ and added back.

   The refinement is needed because the poloidal spectrum of a mode sits near $m\approx nq$, which can be above the
   Nyquist limit `gridNy`/2 of the coarse grid, while the field-aligned envelope is smooth and interpolates well.
   `{0, 0, 0}` isolates the zonal $(n, m) = (0, 0)$ component.

## Initial perturbation

Only $\Phi$ is initialized, on the interior nodes; every other field and all marker weights start at zero:

$$
\Phi = a\,e^{-(x - x_c)^2/(2\sigma^2)}\left\{\cos(My - Nz)
  + \sum_{n} \cos\left[m\left(y + \tfrac{2\pi m}{M}\right) - n\left(z + \tfrac{2\pi n}{N}\right)\right]\right\},
$$

with $a$ = `perturbAmplitude`, $\sigma$ = `perturbWidth`, $x_c = ($`perturbRadialIndex`$-1)\Delta x$,
$N$ = `perturbRightN`·`tubes`, and the sum over $n$ from `perturbLeftN`·`tubes` to (`perturbRightN`−1)·`tubes` in steps
of `tubes`. The poloidal numbers are resonant, $M = \mathrm{round}(N\tilde q)$ and $m = \mathrm{round}(n\tilde q)$,
with $\tilde q = q(x_c)$ rounded to 0.1. The vorticity is then computed from $\Phi$.

A restart reads `MHDContinue_<step>.bin` (the eleven fields $\Phi$, $A$, $\delta n_e$, $\delta T_e$, $\delta p$,
$\delta p_{i,a,b}$, $\delta n_{i,a,b}$) and `PICContinue_<step>.bin` (`partConst`, keys and markers).
