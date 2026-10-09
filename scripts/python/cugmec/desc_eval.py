"""
Evaluate a DESC equilibrium on the cuGMEC grid: the contents of standard2D.mat, as a dict.

The helper functions below are copied from scripts/equilibrium/output2D_0170.ipynb (cell 2.1), changing only whitespace, so they can be
compared line by line. standard2d() is outputStandard2D() from the same cell, returning the dict instead of writing
standard2D.mat. It computes psi_p for any iota profile, not only a PowerSeriesProfile.
"""
import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline

import desc
import desc.grid
from desc.profiles import PowerSeriesProfile


# ---------------------------------------------------------------- copied from output2D_0170.ipynb, cell 2.1

def getRhoThetaPEST(rho_num, rho_min, rho_max, theta_num):

    rho_grid = np.linspace(rho_min, rho_max, rho_num)

    dtheta_pest = 2*np.pi/theta_num
    theta_pest_min = -np.pi + dtheta_pest/2
    theta_pest_max =  np.pi - dtheta_pest/2
    theta_pest = np.linspace(theta_pest_min, theta_pest_max, theta_num)

    return rho_grid, theta_pest

def uniqueList(values):
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out

def scalarField(field, name, shape):
    return np.asarray(field[name]).reshape(shape)

def vectorField(field, name, shape):
    return np.asarray(field[name]).reshape(shape + (3,))

def dotVec(a, b):
    return np.sum(a * b, axis=-1)

def makeNodes2D(rho_values, theta_values):
    rr, tt = np.meshgrid(rho_values, theta_values, indexing='ij')
    zz = np.zeros_like(rr)
    return np.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=1)

def linearCombination(coeffs, bases, shape):
    out = np.zeros(shape + (3,))
    for coeff, basis in zip(coeffs, bases):
        out += coeff[..., None] * basis
    return out

def derivativeCombination(weights, derivs, shape):
    out = np.zeros(shape + (3,))
    for weight, deriv in zip(weights, derivs):
        out += weight[..., None] * deriv
    return out

def addDescMetricValues2D(field, values, shape):
    desc_con = {
        'rr': 'g^rr', 'rt': 'g^rt', 'rz': 'g^rz',
        'tt': 'g^tt', 'tz': 'g^tz', 'zz': 'g^zz',
    }
    for comp, var in desc_con.items():
        values[f'gcon_{comp}'] = scalarField(field, var, shape)
        values[f'dgcon_{comp}_dr'] = scalarField(field, f'{var}_r', shape)
        values[f'dgcon_{comp}_dt'] = scalarField(field, f'{var}_t', shape)

    desc_cov = {
        'rr': 'g_rr', 'rt': 'g_rt', 'rz': 'g_rz',
        'tt': 'g_tt', 'tz': 'g_tz', 'zz': 'g_zz',
    }
    for comp, var in desc_cov.items():
        values[f'gcov_{comp}'] = scalarField(field, var, shape)

    e = {
        'r': vectorField(field, 'e_rho', shape),
        't': vectorField(field, 'e_theta', shape),
        'z': vectorField(field, 'e_zeta', shape),
    }
    de = {
        'r': {
            'r': vectorField(field, 'e_rho_r', shape),
            't': vectorField(field, 'e_rho_t', shape),
            'z': vectorField(field, 'e_rho_z', shape),
        },
        't': {
            'r': vectorField(field, 'e_rho_t', shape),
            't': vectorField(field, 'e_theta_t', shape),
            'z': vectorField(field, 'e_theta_z', shape),
        },
    }

    metric_list = [('rr','r','r'), ('rt','r','t'), ('rz','r','z'),
                   ('tt','t','t'), ('tz','t','z'), ('zz','z','z')]
    for comp, i, j in metric_list:
        for coord in ['r','t']:
            values[f'dgcov_{comp}_d{coord}'] = dotVec(de[coord][i], e[j]) + dotVec(e[i], de[coord][j])

def pestTransformCoefficients2D(field, shape):
    lr = scalarField(field, 'lambda_r', shape)
    lt = scalarField(field, 'lambda_t', shape)
    lrr = scalarField(field, 'lambda_rr', shape)
    lrt = scalarField(field, 'lambda_rt', shape)
    ltt = scalarField(field, 'lambda_tt', shape)

    one = np.ones(shape)
    zero = np.zeros(shape)
    D = one + lt

    a = -lr / D
    b = one / D
    c = zero

    T = {
        'x': [one, a, zero],
        'y': [zero, b, zero],
    }

    lr_m = {'r': lrr, 't': lrt}
    lt_m = {'r': lrt, 't': ltt}
    lz_m = {'r': zero, 't': zero}

    da = {m: -(lr_m[m] * D - lr * lt_m[m]) / D**2 for m in ['r','t']}
    db = {m: -lt_m[m] / D**2 for m in ['r','t']}
    dc = {m: zero for m in ['r','t']}

    cov_C = {
        'x': [one, a, zero],
        'y': [zero, b, zero],
        'z': [zero, c, one],
    }
    cov_dC_old = {
        m: {
            'x': [zero, da[m], zero],
            'y': [zero, db[m], zero],
            'z': [zero, dc[m], zero],
        } for m in ['r','t']
    }

    con_L = {
        'x': [one, zero, zero],
        'y': [lr, D, zero],
        'z': [zero, zero, one],
    }
    con_dL_old = {
        m: {
            'x': [zero, zero, zero],
            'y': [lr_m[m], lt_m[m], lz_m[m]],
            'z': [zero, zero, zero],
        } for m in ['r','t']
    }

    return T, cov_C, cov_dC_old, con_L, con_dL_old

def buildNewBasis2D(basis_old, dbasis_old, T, coeffs, dcoeffs_old, shape):
    old_coords = ['r','t','z']
    old_deriv_coords = ['r','t']
    new_coords = ['x','y','z']
    deriv_coords = ['x','y']

    basis = {}
    deriv = {coord: {} for coord in deriv_coords}

    for name in new_coords:
        basis[name] = linearCombination(coeffs[name], basis_old, shape)

    for deriv_coord in deriv_coords:
        for basis_name in new_coords:
            dcoeff = []
            for old_i in range(3):
                temp = np.zeros(shape)
                for weight, old_coord in zip(T[deriv_coord], old_coords):
                    if old_coord in old_deriv_coords:
                        temp += weight * dcoeffs_old[old_coord][basis_name][old_i]
                dcoeff.append(temp)

            out = linearCombination(dcoeff, basis_old, shape)
            for coeff, old_i in zip(coeffs[basis_name], range(3)):
                old_derivs = [dbasis_old[old_coord][old_i] for old_coord in old_deriv_coords]
                out += coeff[..., None] * derivativeCombination(T[deriv_coord][0:2], old_derivs, shape)
            deriv[deriv_coord][basis_name] = out

    return basis, deriv

def addPestMetricValues2D(field, values, shape):
    T, cov_C, cov_dC_old, con_L, con_dL_old = pestTransformCoefficients2D(field, shape)

    pest_cov_direct = {
        'xx': 'g_rr|PEST', 'xy': 'g_rv|PEST', 'xz': 'g_rp|PEST',
        'yy': 'g_vv|PEST', 'yz': 'g_vp|PEST', 'zz': 'g_pp|PEST',
    }
    for comp, var in pest_cov_direct.items():
        values[f'PESTcov_{comp}'] = scalarField(field, var, shape)

    cov_old = [
        vectorField(field, 'e_rho', shape),
        vectorField(field, 'e_theta', shape),
        vectorField(field, 'e_zeta', shape),
    ]
    dcov_old = {
        'r': [
            vectorField(field, 'e_rho_r', shape),
            vectorField(field, 'e_rho_t', shape),
            vectorField(field, 'e_rho_z', shape),
        ],
        't': [
            vectorField(field, 'e_rho_t', shape),
            vectorField(field, 'e_theta_t', shape),
            vectorField(field, 'e_theta_z', shape),
        ],
    }
    pest_cov_basis, pest_cov_deriv = buildNewBasis2D(cov_old, dcov_old, T, cov_C, cov_dC_old, shape)

    metric_list = [('xx','x','x'), ('xy','x','y'), ('xz','x','z'),
                   ('yy','y','y'), ('yz','y','z'), ('zz','z','z')]
    for comp, i, j in metric_list:
        for coord in ['x','y']:
            suffix = f'p{coord}'
            values[f'dPESTcov_{comp}_{suffix}'] = dotVec(pest_cov_deriv[coord][i], pest_cov_basis[j]) + dotVec(pest_cov_basis[i], pest_cov_deriv[coord][j])

    pest_cov_direct_deriv = {
        ('xx','py'): '(g_rr_v)|PEST',
        ('yy','px'): '(g_vv_r)|PEST',
        ('zz','py'): '(g_pp_v)|PEST',
    }
    for (comp, suffix), var in pest_cov_direct_deriv.items():
        if var in field:
            values[f'dPESTcov_{comp}_{suffix}'] = scalarField(field, var, shape)

    pest_con_direct = {
        'xx': 'g^rr', 'xy': 'g^rv', 'xz': 'g^rz', 'zz': 'g^zz',
    }
    for comp, var in pest_con_direct.items():
        values[f'PESTcon_{comp}'] = scalarField(field, var, shape)

    con_old = [
        vectorField(field, 'e^rho', shape),
        vectorField(field, 'e^theta', shape),
        vectorField(field, 'e^zeta', shape),
    ]
    dcon_old = {
        'r': [
            vectorField(field, 'e^rho_r', shape),
            vectorField(field, 'e^theta_r', shape),
            vectorField(field, 'e^zeta_r', shape),
        ],
        't': [
            vectorField(field, 'e^rho_t', shape),
            vectorField(field, 'e^theta_t', shape),
            vectorField(field, 'e^zeta_t', shape),
        ],
    }
    pest_con_basis, pest_con_deriv = buildNewBasis2D(con_old, dcon_old, T, con_L, con_dL_old, shape)

    values['PESTcon_yy'] = dotVec(pest_con_basis['y'], pest_con_basis['y'])
    values['PESTcon_yz'] = dotVec(pest_con_basis['y'], pest_con_basis['z'])

    for comp, i, j in metric_list:
        for coord in ['x','y']:
            suffix = f'p{coord}'
            values[f'dPESTcon_{comp}_{suffix}'] = dotVec(pest_con_deriv[coord][i], pest_con_basis[j]) + dotVec(pest_con_basis[i], pest_con_deriv[coord][j])

    pest_con_direct_deriv = {
        ('xx','py'): '(g^rr_v)|PEST',
        ('xy','py'): '(g^rv_v)|PEST',
        ('xz','py'): '(g^rz_v)|PEST',
    }
    for (comp, suffix), var in pest_con_direct_deriv.items():
        if var in field:
            values[f'dPESTcon_{comp}_{suffix}'] = scalarField(field, var, shape)

def pestMetricComputeList2D():
    direct = [
        'lambda_r','lambda_t','lambda_rr','lambda_rt','lambda_tt',
        'e_rho','e_theta','e_zeta','e_theta_PEST','e_rho_r','e_rho_t','e_rho_z','e_theta_t','e_theta_z',
        'e^rho','e^theta','e^zeta','e^rho_r','e^rho_t','e^theta_r','e^theta_t','e^zeta_r','e^zeta_t',
        'g_rr|PEST','g_rv|PEST','g_rp|PEST','g_vv|PEST','g_vp|PEST','g_pp|PEST',
        'g^rr','g^rv','g^rz','g^zz',
        '(g_rr_v)|PEST','(g_vv_r)|PEST','(g_pp_v)|PEST',
        '(g^rr_v)|PEST','(g^rv_v)|PEST','(g^rz_v)|PEST',
    ]
    return uniqueList(direct)

def standardComputeList2D():
    direct = [
        'R','Z',
        'sqrt(g)',
        'sqrt(g)_PEST','(sqrt(g)_PEST_r)|PEST','(sqrt(g)_PEST_v)|PEST',
        'J_parallel',
        '|B|','|B|_r','|B|_t','|B|_rr','|B|_rt','|B|_tt',
        'lambda','lambda_r','lambda_t','lambda_rr','lambda_rt','lambda_tt',
        'rho','q','iota','iota_r','iota_rr','theta',
        'e_rho','e_theta','e_zeta','e_rho_r','e_rho_t','e_rho_z','e_theta_t','e_theta_z',
    ]
    for var in ['g^rr','g^rt','g^rz','g^tt','g^tz','g^zz']:
        direct.append(var)
        direct.append(f'{var}_r')
        direct.append(f'{var}_t')
    direct += ['g_rr','g_rt','g_rz','g_tt','g_tz','g_zz']
    direct += pestMetricComputeList2D()
    return uniqueList(direct)

def collectStandardValues2D(field, shape):
    values = {}

    values['R'] = scalarField(field, 'R', shape)
    values['Z'] = scalarField(field, 'Z', shape)
    values['JDESC'] = scalarField(field, 'sqrt(g)', shape)
    values['JPEST'] = scalarField(field, 'sqrt(g)_PEST', shape)
    values['dJPEST_drho'] = scalarField(field, '(sqrt(g)_PEST_r)|PEST', shape)
    values['dJPEST_dtheta'] = scalarField(field, '(sqrt(g)_PEST_v)|PEST', shape)
    values['J_parallel'] = scalarField(field, 'J_parallel', shape)

    b_map = {
        'B': '|B|', 'dB_dr': '|B|_r', 'dB_dt': '|B|_t',
        'dB_dr2': '|B|_rr', 'dB_drdt': '|B|_rt', 'dB_dt2': '|B|_tt',
    }
    for out_name, desc_name in b_map.items():
        values[out_name] = scalarField(field, desc_name, shape)

    lambda_map = {
        'lambda': 'lambda', 'dlambda_dr': 'lambda_r', 'dlambda_dt': 'lambda_t',
        'dlambda_dr2': 'lambda_rr', 'dlambda_drdt': 'lambda_rt', 'dlambda_dt2': 'lambda_tt',
    }
    for out_name, desc_name in lambda_map.items():
        values[out_name] = scalarField(field, desc_name, shape)

    addDescMetricValues2D(field, values, shape)
    addPestMetricValues2D(field, values, shape)

    e_rho = vectorField(field, 'e_rho', shape)
    e_theta = vectorField(field, 'e_theta', shape)
    lambda_r = scalarField(field, 'lambda_r', shape)
    lambda_t = scalarField(field, 'lambda_t', shape)
    e_rho_pest = e_rho - (lambda_r / (1.0 + lambda_t))[..., None] * e_theta
    values['dR_drho'] = e_rho_pest[..., 0]
    values['dZ_drho'] = e_rho_pest[..., 2]

    e_theta_pest = vectorField(field, 'e_theta_PEST', shape)
    values['dR_dtheta'] = e_theta_pest[..., 0]
    values['dZ_dtheta'] = e_theta_pest[..., 2]

    for name in ['rho','q','iota','iota_r','iota_rr']:
        values[name] = scalarField(field, name, shape)

    return values


# ---------------------------------------------------------------- driver (outputStandard2D)

def poloidal_flux(eq, rho, psit_max):
    """psi_p(rho) and dpsi_p/drho = iota * dpsi_t/drho, with psi_t = psit_max * rho^2 (Wb/rad)."""
    if isinstance(eq.iota, PowerSeriesProfile):  # closed form, as in the notebook
        orders, coeffs = eq.iota.basis.modes[:, 0], eq.iota.params
        index = np.zeros((rho.size, 3))
        index[:, 0] = rho
        grid = desc.grid.Grid(index)
        psip = PowerSeriesProfile(params=coeffs / (orders + 2) * 2 * psit_max, modes=orders + 2).compute(grid)
        dpsip = PowerSeriesProfile(params=coeffs * 2 * psit_max, modes=orders + 1).compute(grid)
        return np.asarray(psip), np.asarray(dpsip)
    fine = np.linspace(0.0, 1.0, 2001)
    iota = _iota(eq, fine)
    integrand = iota * 2 * psit_max * fine
    psip = CubicSpline(fine, cumulative_simpson(integrand, x=fine, initial=0.0))(rho)
    return psip, _iota(eq, rho) * 2 * psit_max * rho


def _iota(eq, rho):
    grid = desc.grid.LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    return grid.compress(eq.compute("iota", grid=grid)["iota"])


def standard2d(eq, rho_min, rho_max, rho_num, theta_num, n_segments=8):
    """Evaluate eq on rho_num uniform rho in [rho_min, rho_max] x theta_num cell-centred theta_PEST. Returns standard2D."""
    if rho_num % n_segments:
        raise ValueError(f"rho_num = {rho_num} is not divisible by n_segments = {n_segments}")
    rho_grid, theta_pest = getRhoThetaPEST(rho_num, rho_min, rho_max, theta_num)
    theta_desc = np.zeros((rho_num, theta_num))
    compute_list = standardComputeList2D()
    outputs = None
    segment_size = rho_num // n_segments

    for seg in range(n_segments):
        start, end = seg * segment_size, (seg + 1) * segment_size
        seg_shape = (segment_size, theta_num)
        grid_index = makeNodes2D(rho_grid[start:end], theta_pest)
        field = eq.compute(compute_list, desc.grid.Grid(grid_index, coordinates='rvp'))
        values = collectStandardValues2D(field, seg_shape)
        if outputs is None:
            outputs = {name: np.zeros((rho_num, theta_num)) for name in values}
        for name, value in values.items():
            outputs[name][start:end, :] = value
        theta_desc[start:end, :] = scalarField(field, 'theta', seg_shape)

    axis = desc.grid.Grid(np.array([0, 0, 0]))
    B00 = np.asarray(eq.compute('|B|', axis)['|B|'])
    L00 = np.asarray(eq.compute('R', axis)['R'])
    psit_max = np.asarray(eq.compute('psi', desc.grid.Grid(np.array([1, 0, 0])))['psi'])
    psip, dpsipdr = poloidal_flux(eq, rho_grid, float(psit_max[0]))

    out = dict(outputs)
    out['diota_dr'] = outputs['iota_r']
    out['diota_dr2'] = outputs['iota_rr']
    out.update(theta_pest=np.repeat(theta_pest[None, :], rho_num, axis=0), theta_desc=theta_desc,
               B00=B00, L00=L00, psit_max=psit_max,
               psip=np.repeat(psip[:, None], theta_num, axis=1), dpsip_dr=np.repeat(dpsipdr[:, None], theta_num, axis=1))
    return out
