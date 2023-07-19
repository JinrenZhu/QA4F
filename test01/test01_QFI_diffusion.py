# This script requires Sympy package, which is not pre-installed in default Leap IDE

# %% prelim
import numpy as np
from sympy import symbols, simplify, latex, expand, diff
from dimod import BinaryQuadraticModel, ExactSolver, generators
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.inspector import show

# %% functional
# binary variables
kb0, kb1, kb2, kb3 = symbols('\kappa^b_0, \kappa^b_1, \kappa^b_2, \kappa^b_3')
kb4, kb5, kb6, kb7 = symbols('\kappa^b_4, \kappa^b_5, \kappa^b_6, \kappa^b_7')
kb8, kb9, kb10, kb11 = symbols('\kappa^b_8, \kappa^b_9, \kappa^b_10, \kappa^b_11')
# encoding
kh0 = 1 / 8 * (kb0 + 2 * kb1 + 4 * kb2 + 8 * kb3 - 7)
kh1 = 1 / 8 * (kb4 + 2 * kb5 + 4 * kb6 + 8 * kb7 - 7)
kh2 = 1 / 8 * (kb8 + 2 * kb9 + 4 * kb10 + 8 * kb11 - 7)

# functions
L1 = lambda x : (-1.0 + x) * (-1.0 + 2.0 * x)
L2 = lambda x : -4.0 * (-1.0 + x) * x
L3 = lambda x : x * (-1.0 + 2.0 * x)
dL1dx = lambda x : -3.0 + 4.0 * x
dL2dx = lambda x : 4.0 - 8.0 * x
dL3dx = lambda x : -1.0 + 4.0 * x

# observation
# total number of observations
n_obs = 4
# location of observations
x_obs = np.linspace(1, n_obs+1, n_obs, endpoint=False) / (n_obs+1)
# x_obs = np.random.rand(n_obs)
# observation
u_obs = np.sin(np.pi * x_obs)
dudx_obs = np.pi * np.cos(np.pi * x_obs)
d2udx2_obs = - np.pi**2 * np.sin(np.pi * x_obs)
f_obs = np.pi * x_obs * (-2.0 * np.cos(np.pi*x_obs)
                         + np.pi * x_obs * np.sin(np.pi*x_obs))

# functional
kh = L1(x_obs) * kh0 + L2(x_obs) * kh1 + L3(x_obs) * kh2
dkhdx = dL1dx(x_obs) * kh0 + dL2dx(x_obs) * kh1 + dL3dx(x_obs) * kh2
res_pde = -dkhdx * dudx_obs - kh * d2udx2_obs - f_obs
obj_func = expand(np.sum(np.square(res_pde)))

# write
with open('./res_sq_latx.txt', 'w') as latex_writer:
    latex_writer.write(latex(obj_func))

# %% covariance matrix
kb = [kb0, kb1, kb2, kb3, kb4, kb5, kb6, kb7, kb8, kb9, kb10, kb11]

obj_func_diag = obj_func
cov_matrix_off = np.zeros([12, 12])
for i in range(0, 12):
    for j in range(i+1, 12):
        cov_matrix_off[i, j] = diff(diff(obj_func, kb[i]), kb[j])
        obj_func_diag -= cov_matrix_off[i, j] * kb[i] * kb[j]

obj_func_diag_linear = obj_func_diag
diag_quad = np.zeros([12])
for i in range(0, 12):
    diag_quad[i] = diff(diff(obj_func_diag, kb[i]), kb[i]) / 2
    obj_func_diag_linear -= kb[i] * kb[i] * diag_quad[i]

obj_func_diag_const = obj_func_diag_linear
diag_linear = np.zeros([12])
for i in range(0, 12):
    diag_linear[i] = diff(obj_func_diag_linear, kb[i])
    obj_func_diag_const -= kb[i] * diag_linear[i]

bias_linear = diag_quad + diag_linear

# Generate QUBO
bqm_diffusion = BinaryQuadraticModel('BINARY')
bqm_diffusion.add_linear_from_array(bias_linear)
bqm_diffusion.add_quadratic_from_dense(cov_matrix_off)

# %% sampling
# sampleset_SA = SimulatedAnnealingSampler().sample(bqm_diffusion, num_reads=4000)
# print(sampleset_SA.first)
# print(len(sampleset_SA.lowest())/len(sampleset_SA))
# np.save('SA_samples', sampleset_SA.record)

# %% quantum annealing
samplerQA = EmbeddingComposite(DWaveSampler())
sampleset_QA = samplerQA.sample(bqm_diffusion, num_reads=2000, chain_strength=50, label='Diffusion- chain')
show(sampleset_QA) 
print(sampleset_QA.first)
print(len(sampleset_QA.lowest())/len(sampleset_QA))
np.save('QA_samples_chain_50', sampleset_QA.record)