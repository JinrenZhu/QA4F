# This script requires Sympy package, which is not pre-installed in default Leap IDE

# %% prelim
import numpy as np
from sympy import symbols, simplify, latex, expand, diff, Array, collect
from scipy import interpolate
from time import perf_counter
import matplotlib.pyplot as plt

from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler

# %% observation: Lee & Moser 2015 Re_\tau = 180
LM_path = './'
mean_raw = np.loadtxt(LM_path + 'LM_Channel_0180_mean_prof.dat',
                     comments='%')
RS_raw = np.loadtxt(LM_path + 'LM_Channel_0180_vel_fluc_prof.dat',
                    comments='%')
y_coord = mean_raw[:, 0]
U = mean_raw[:, 2]
dUdy = mean_raw[:, 3]
uu = RS_raw[:, 2]
uv = RS_raw[:, 5]

# interpolate: cubic bspline
dUdy_interp = interpolate.CubicSpline(y_coord, dUdy)
d2Udy2 = dUdy_interp.derivative(1)

# %% binary variables
# 12 real varialbes * 10 binary encoding = 120 variables
n_real = 12
n_encode = 10
n_bin = n_real * n_encode
range_encode = 30

# variable name expression
var_bin = np.chararray([n_real, n_encode], itemsize=9)
for ireal in range(0, n_real):
    for ibin in range(0, n_encode):
        var_bin[ireal, ibin] = 'nu' + '^' + str(ibin).zfill(2) + '_' + str(ireal+1).zfill(2)

# create symbolic variables
nu_t_bin_var = []
for ireal in range(0, n_real):
    nu_t_bin_var_temp = []
    for ibin in range(0, n_encode):
        nu_t_bin_var_temp.append(symbols(var_bin[ireal, ibin].decode('ASCII')))
    nu_t_bin_var.append(nu_t_bin_var_temp)

nu_t_bin_var_1D = []
for ireal in range(0, n_real):
    for ibin in range(0, n_encode):
        nu_t_bin_var_1D.append(nu_t_bin_var[ireal][ibin])
        
# binary encoding: 0~20
nu_t_real_var = []
nu_t_real_var.append(0)
for ireal in range(0, n_real):
    nu_t_real_var_ireal = 0
    for ibin in range(0, n_encode):
        nu_t_real_var_ireal += 2**ibin * nu_t_bin_var[ireal][ibin]
    nu_t_real_var.append(nu_t_real_var_ireal/(2**n_encode)*range_encode)

# %% Forward problem
# FEM discretization
nel = 6
nen = nel * 2 + 1
y_max = 1.0
node_coord = np.linspace(0, y_max, nen)
elmt_conn = np.zeros([nel, 3], dtype=int)
for iel in range(0, nel):
    elmt_conn[iel, :] = np.array([iel*2, iel*2+1, iel*2+2])

# shape functions in natural coordinates [-1, 1]
L0 = lambda x : 0.5 * (-1.0 + x) * x
L1 = lambda x : (1.0 - x) * (1.0 + x)
L2 = lambda x : 0.5 * x * (1.0 + x)
dL0dx = lambda x : - 0.5 + x
dL1dx = lambda x : -2.0 * x
dL2dx = lambda x : 0.5 + x

# FEM interpolation
def nu_t_FEM(y_loc, nu_t_full):
    nu_t = []
    nu_t_grad = []
    # go over all locations
    for iy in range(0, y_loc.shape[0]):
        y_loc_i = y_loc[iy]
        # determine the element
        iel = np.asarray(node_coord[elmt_conn[:,-1]] >= y_loc_i).nonzero()[0][0]
        conn_i = elmt_conn[iel, :]
        nu_t_h_i_0 = nu_t_full[conn_i[0]]
        nu_t_h_i_1 = nu_t_full[conn_i[1]]
        nu_t_h_i_2 = nu_t_full[conn_i[2]]
        # convert to natural coordinates
        coord_i = node_coord[conn_i]
        xi_i = (y_loc_i - coord_i[0]) / (coord_i[-1] - coord_i[0]) * 2.0 - 1.0
        Jcb = 2.0 / (coord_i[-1] - coord_i[0])
        # interpolate fields
        nu_t.append(L0(xi_i) * nu_t_h_i_0 +
                    L1(xi_i) * nu_t_h_i_1 +
                    L2(xi_i) * nu_t_h_i_2)
        nu_t_grad.append((dL0dx(xi_i) * nu_t_h_i_0 +
                          dL1dx(xi_i) * nu_t_h_i_1 +
                          dL2dx(xi_i) * nu_t_h_i_2) * Jcb)
    return np.array(nu_t), np.array(nu_t_grad)

# %% Object functional
tic = perf_counter()
# flags
smooth_flag = False
wbc_flg = False

# Observation
n_obs = 10
y_obs = np.linspace(y_max/nel/4, y_max-y_max/nel/4, n_obs)

dUdy_obs = dUdy_interp(y_obs)
d2Udy2_obs = d2Udy2(y_obs)

# FEM estimation
nu_t_obs, nu_t_grad_obs = nu_t_FEM(y_obs, nu_t_real_var)
nu_t_smth, nu_t_grad_smth = nu_t_FEM(node_coord, nu_t_real_var)

# compute functional
res_obs = nu_t_grad_obs * dUdy_obs + (nu_t_obs + 1.0) * d2Udy2_obs + 1.0

obj_func = np.sum(np.square(res_obs))
if smooth_flag:
    smth_penalty = 6e-5
    obj_func += np.sum(np.square(nu_t_grad_smth)) * smth_penalty

obj_func = expand(obj_func)

toc = perf_counter()
print('Analytical object functional: done ... ' + str(toc-tic) + ' sec.s')

# %% QUBO coefficient
tic = perf_counter()

cov_matrix_off = np.zeros([n_bin, n_bin])
diag_quad = np.zeros([n_bin])
diag_linear = np.zeros([n_bin])

obj_func_diag = obj_func
for i in range(0, n_bin):
    for j in range(i+1, n_bin):
        cov_matrix_off[i, j] = obj_func.coeff(nu_t_bin_var_1D[i]*nu_t_bin_var_1D[j], 1)
        obj_func_diag -= cov_matrix_off[i, j] * nu_t_bin_var_1D[i] * nu_t_bin_var_1D[j]
        
obj_func_diag_linear = obj_func_diag
for i in range(0, n_bin):
    diag_quad[i] = obj_func_diag.coeff(nu_t_bin_var_1D[i], 2)
    obj_func_diag_linear -= nu_t_bin_var_1D[i] * nu_t_bin_var_1D[i] * diag_quad[i]
    
obj_func_diag_const = obj_func_diag_linear
for i in range(0, n_bin):
    diag_linear[i] = obj_func_diag_linear.coeff(nu_t_bin_var_1D[i], 1)
    obj_func_diag_const -= nu_t_bin_var_1D[i] * diag_linear[i]
    
bias_linear = diag_quad + diag_linear

toc = perf_counter()
print('QUBO coefficients: done ... ' + str(toc-tic) + ' sec.s')

# %% sampling
bqm_diffusion = BinaryQuadraticModel('BINARY')
bqm_diffusion.add_linear_from_array(bias_linear)
bqm_diffusion.add_quadratic_from_dense(cov_matrix_off)

# random sampler
sampleset_SA = SimulatedAnnealingSampler().sample(bqm_diffusion, num_reads=10000)
print(sampleset_SA.first)
print(len(sampleset_SA.lowest())/len(sampleset_SA))
np.save('SA_samples', sampleset_SA.record)
