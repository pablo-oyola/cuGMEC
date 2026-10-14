# Field solvers

## Poisson equation

The gyrokinetic Poisson equation {eq}`eq-poisson`, $\varpi = \nabla_\perp\cdot(V_A^{-2}\nabla_\perp\Phi)$, is solved for
$\Phi$ after every Runge–Kutta stage, independently on each $y$ plane:

- **Matrix.** One sparse matrix of size $(N_x N_z)^2$ per plane, built on the host. It discretizes the operator in
  $(x, z)$ with the full metric ($g^{xx}$, $g^{xz}$, $g^{zz}$): interior rows use a $5\times5$ stencil (fourth-order
  differences in $x$ and $z$, the mixed term as a product of first derivatives), rows $i = 1$ and $N_x - 2$ use
  one-sided stencils in $x$, and $z$ is periodic. No Fourier transform is used, so all toroidal and poloidal couplings
  of the metric are kept. The matrix differs between planes because the metric depends on $\theta$ (and on $z$ in a
  stellarator).
- **Boundary rows.** Rows $i = 0$ and $N_x - 1$ are identity rows (Dirichlet), and $\varpi$ is zero there, so
  $\Phi = 0$ on both walls.
- **Solver.** The matrices are assembled in COO, converted to CSR with cuSPARSE, and factorized once at start-up with
  NVIDIA cuDSS (general matrix type, in `mhdReal` precision), one handle and one CUDA stream per plane. Each stage only
  runs the cuDSS solve phase with the new $\varpi$. This replaces the CPU PARDISO solver of GMEC.

With `ifFLRMHD`, the solution is corrected afterwards, $\Phi = \Phi_0 - \rho_i^2V_A^2\varpi$
([](../physics/fluid.md)).

## Ampère's law

No solve is needed: $A$ is evolved, and $\delta J_\parallel/B$ is evaluated explicitly from it after every stage
{eq}`eq-ampere`.

## Implicit dissipation

The same machinery (matrix per plane, Dirichlet rows, cuDSS factorization at start-up) builds the operators of the
implicit dissipation, applied after each step by solving $M f^{n+1} = f^n$:

| Matrix | Operator | Fields |
|---|---|---|
| `Perp2Phi`, `Perp2dNe`, `Perp2dTe`, `Perp2dP` | $M = I - \Delta t\,\nu_\perp\nabla_\perp^2$ | $\Phi$, $\delta n_e$, $\delta T_e$, $\delta p$; `Perp2dP` also on the kinetic $\delta p_s$, `Perp2dNe` on $\delta n_s$ |
| `Resistive` | $M = I + \Delta t\,\eta\,B\,\mathcal L_J$, with $\mathcal L_J$ the operator $A\mapsto\delta J_\parallel/B$ | $A$: a backward-Euler step of $\partial_t A = -\eta\,\delta J_\parallel$ |

Each is a backward-Euler step, unconditionally stable. Note that `ifNablaPerp2A`/`perp2A`, listed as perpendicular
dissipation of $A$, is physically a parallel resistivity.
