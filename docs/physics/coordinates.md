# Coordinates, geometry and normalization

## Flux coordinates

cuGMEC works in straight-field-line (PEST) flux coordinates $(\rho, \theta, \varphi)$ computed from a DESC equilibrium
(or, with the Python tools, traced directly from a gEQDSK file with pyCOCOS):

- $\rho = \sqrt{\psi_t/\psi_{t,b}}$ is the square root of the normalized toroidal flux, with $\psi_t = \psi_{t,b}\rho^2$
  and $\psi_{t,b}/2\pi$ = `PSITMAX` (Wb/rad);
- $\theta$ is the PEST poloidal angle, in which field lines are straight: $\mathbf B\cdot\nabla\varphi = q\,\mathbf B\cdot\nabla\theta$;
- $\varphi$ is the toroidal angle.

The grid uses three computational coordinates:

| Coordinate | Definition | Grid |
|---|---|---|
| $x$ | $x = (\rho - \rho_0)/\Delta\rho$, $\Delta\rho = \rho_1 - \rho_0$ (`RHO0`, `RHO1`) | `gridNx` nodes on $[0, 1]$, both ends included |
| $y$ | $\theta$ | `gridNy` cell centres on $[-\pi, \pi)$, two ghost planes on each side |
| $z$ | $\varphi$ over a wedge of $2\pi/$`tubes` | `gridNz` cell centres on $[-\pi/\mathrm{tubes}, \pi/\mathrm{tubes})$, periodic |

The radial domain is an annulus, $\rho\in[\rho_0, \rho_1]$ with $\rho_0 > 0$: the magnetic axis is never in the domain.
The wedge holds the toroidal mode numbers $n = \mathrm{tubes}\cdot\ell$, and the retained modes are
$\ell\in[$`leftN`, `rightN`$]$. For a stellarator the preprocessing sets `tubes` = `NFP` (one field period) or 1.

$y$ is the direction along the field: `gridNy` is small (32 in the examples), because the perturbations vary slowly
along $\mathbf B$, and it is the decomposed direction ([](../numerics/parallel.md)).

## Shifted metric

A field-aligned coordinate $\alpha = \varphi - q\theta$ makes the parallel direction a grid direction, but its metric
grows secularly with $\theta$ through $q'\theta$, which deforms the grid cells. cuGMEC uses the shifted metric
{cite:p}`scott2001,jiang2024gmec1` instead. Around each poloidal plane $y_j = \theta_j$ the local coordinates are

$$
x,\qquad y = \theta,\qquad z' = \varphi - q(\rho)(\theta - \theta_j),
$$

so that on the plane itself $z' = \varphi$ and

$$
\nabla x = \nabla\rho/\Delta\rho,\qquad \nabla y = \nabla\theta,\qquad \nabla z' = \nabla\varphi - q\nabla\theta .
$$

Every MHD field is stored in this **shifted representation**: on each plane $y_j$ the third index is the toroidal angle
$\varphi$ itself, so the fields are simply periodic in $\theta$. Because $\mathbf B\cdot\nabla z' = \mathbf B\cdot\nabla x = 0$,
the field has a single contravariant component,

$$
\mathbf B = B^y\mathbf e_y,\qquad B^y = \mathbf B\cdot\nabla\theta = \frac{\psi_p'(\rho)}{\sqrt g},
$$

and its covariant components are $B_i = g_{iy}B^y$. The metric of the local frame (the "SFT" metric) has no secular term:

$$
\begin{aligned}
g^{xx} &= g^{\rho\rho}/\Delta\rho^2, & g^{xy} &= g^{\rho\theta}/\Delta\rho, & g^{xz} &= (g^{\rho\varphi} - q g^{\rho\theta})/\Delta\rho,\\
g^{yy} &= g^{\theta\theta}, & g^{yz} &= g^{\theta\varphi} - q g^{\theta\theta}, & g^{zz} &= g^{\varphi\varphi} - 2q g^{\theta\varphi} + q^2 g^{\theta\theta}.
\end{aligned}
$$

This is the "simple" shifted metric of {cite:t}`jiang2024gmec1` ($z' = \varphi$ on each plane, $g^{xz}\neq0$); the
orthogonalizing variant of that paper is not used. The Jacobian $J = (\nabla x\times\nabla y\cdot\nabla z)^{-1} =
\Delta\rho\sqrt{g_\mathrm{PEST}}$ is the same in the shifted and the aligned coordinates.

**Parallel derivative.** The derivative along $\mathbf B$ at fixed field line is taken across planes: to differentiate
at plane $j$, the field on planes $j\pm1$, $j\pm2$ is interpolated in $\varphi$ to the points
$\varphi + q[\theta_{j\pm p} - \theta_j]$ (four-point Lagrange), and the fourth-order central difference is applied:

$$
\mathbf b\cdot\nabla f = \frac{B^y}{B}\,\partial_y f\big|_\alpha .
$$

**Aligned representation.** Fields handed to the particles are converted to the field-aligned coordinate,
$f_\mathrm{al}(x, y, \alpha) = f(x, y, \varphi = \alpha + q\theta)$, by the same interpolation; deposited moments go back
the other way. In this representation the field lines close with the twist-shift condition
$f(x, y + 2\pi, \alpha) = f(x, y, \alpha + 2\pi q)$.

## Equilibrium data

The equilibrium file (`MHDEquilibrium`, headerless double precision; HDF5 with the Python tools) holds, on the
$(x, y)$ grid (and $z$ in 3D), the 159 blocks of [](../guide/python.md):

- the shift $q\theta$ on the ghosted $y$ grid;
- the shifted metric (`gcon*`, `gcov*`) and the global field-aligned metric of $(x, \theta, \alpha)$ (`SFAcon*`,
  `SFAcov*`, used by the particles), with their first derivatives;
- $B^y$ (`Bny`), $|B|$ with its first and second derivatives, the Jacobian $J$, $J_{\parallel0}/B$ (`JpB`), the local
  Alfvén speed `Va`, the thermal-ion Larmor radius `Rho`, $R$, $Z$, $\psi_p$ and $q$;
- the profiles $n_i$, $T_i$, $p_i$, $n_e$, $T_e$, $p_e$, $n_\alpha$, $T_\alpha$, $n_b$, $T_b$ and their radial
  derivatives, as flux functions.

Derivative blocks follow the convention: `_px` $= \Delta\rho\,\partial_\rho$, `_py` $= \partial_\theta + q\partial_\varphi$
(along the field in the local frame), `_pz` $= \partial_\varphi$.

The curvature is not stored. It is rebuilt from the metric with $b_i = B^y g_{iy}/B$:

$$
(\nabla\times\mathbf b)^i = J^{-1}\epsilon^{ijk}\partial_j b_k,\qquad
\mathbf b\times\boldsymbol\kappa = \nabla\times\mathbf b - \mathbf b\,(\mathbf b\cdot\nabla\times\mathbf b).
$$

### Tokamak and stellarator paths

The path is chosen by `NFP` alone.

- **`NFP = 1` (tokamak):** the equilibrium is axisymmetric and the coefficient arrays depend on $(x, y)$ only; all
  $\partial_\varphi$ of equilibrium quantities are dropped. The particles use the field-aligned ("Aligned") kernels.
- **`NFP` $\neq 1$ (stellarator):** the equilibrium is sampled on the `gridNz` planes of one field period, the
  coefficients depend on $(x, y, z)$ and keep every $\partial_z$ term, and the particles use the "Straight" kernels in
  $(x, \theta, \varphi)$. The coordinates are still PEST, not Boozer.

:::{warning}
The 3D preprocessing (`writeEquilibrium3D.m`) writes $q$ into the $\psi_p$ slot of the file, so particle quantities
that use $\psi_p$ (the canonical momentum $P_\varphi$, the radial gradient of $f_0$) are wrong in 3D runs made with it.
:::

## Normalization

The reference quantities are computed by the preprocessing (`writeNormalization2D.m`, or the Python tools) and copied
into the parameter file.

| Quantity | Unit |
|---|---|
| magnetic field | $B_0 = \lvert\mathbf B\rvert$ on the magnetic axis |
| length | $L_0 = R$ of the magnetic axis |
| velocity | $v_{A0} = B_0/\sqrt{\mu_0 A_i m_p n_{i,\mathrm{ref}}}$, $n_{i,\mathrm{ref}}$ the first sample of the ion density profile (usually $\rho = 0$), $A_i$ = `IonMass` |
| time | $t_0 = L_0/v_{A0}$; `dt` is in $t_0$ |
| potential, $A_\parallel$, $E$ | $B_0 v_{A0} L_0$, $B_0 L_0$, $v_{A0}B_0$ |
| current density | $B_0/(\mu_0 L_0)$; `JpB` holds $\mu_0 L_0 J_\parallel/B$ |
| number density | $L_0^{-3}$ (the arrays hold $n L_0^3$) |
| pressure | $P_0 = B_0^2/(2\mu_0)$, so a normalized pressure is a $\beta$ |
| electron temperature | $P_0 L_0^3$, so that $\hat n_e\hat T_e = \hat p_e$ |
| charge | $B_0 L_0^2/(\mu_0 v_{A0})$, so the elementary charge is $\hat e = e\mu_0 v_{A0}/(B_0 L_0^2)$ |
| particle velocity, mass, charge | $v_{A0}$, $m_p$, $e$ |
| kinetic temperatures | keV (Maxwellian); the "T" slot of a slowing-down species holds $v_c/v_{A0}$ |

Because the pressure unit is $B_0^2/2\mu_0$, normalized equations carry an explicit factor $1/2$ wherever a pressure or
temperature meets the charge (for instance $\hat T_e/(2\hat e\hat n_e)$ in Ohm's law). The equations on the next pages
are written in SI form, where this factor does not appear.

**Betas.** `IonBeta`, `AlphaBeta` and `BeamBeta` are $\beta_s = p_s/(B_0^2/2\mu_0)$ evaluated at the inner boundary
$\rho = \rho_0$ (the parameter table calls it "on-axis"). The code scales the marker weights so that the deposited
equilibrium pressure at $x = 0$, averaged over $y$ and $z$, equals $\beta_s$ ([](particles.md)).
