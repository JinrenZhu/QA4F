# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:53:14 2023

@author: DELL
"""

# %% prelim
import numpy as np
from time import perf_counter

from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler, PlanarGraphSolver, TreeDecompositionSolver
from dimod import variables

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Computer Modern Roman')
plt.rcParams.update({'font.size':10.5})
plt.rcParams.update({'font.style': 'normal'})
plt.rcParams.update({'font.weight': 'normal'})

# %% parameters
# FEM
nel = 8
nen = nel + 1
# encode
n_real = nen - 2
n_encode = 6
n_bin = n_real * n_encode
# Sampler
SA_n_reads = 50000
SA_sample_rtol = 1.0e-4
# suffix
sffx_1 = str(n_real).zfill(2)
sffx_2 = str(n_encode).zfill(2)
sffx_3 = str(SA_n_reads).zfill(5)

# %% load QUBO coefficient
cov_matrix_off = np.load('cov_mat_off_' + sffx_1 + '_' + sffx_2 + '.npy')
bias_linear = np.load('bias_linear_' + sffx_1 + '_' + sffx_2 + '.npy')

# %% sampling
tic = perf_counter()

# generate QUBO
bqm_diffusion = BinaryQuadraticModel('BINARY')
bqm_diffusion.add_linear_from_array(bias_linear)
bqm_diffusion.add_quadratic_from_dense(cov_matrix_off)

# random sampler
sampleset_SA = SimulatedAnnealingSampler().sample(bqm_diffusion,
                                                  num_reads=SA_n_reads)

# Planer sampler
sampleset_TreeDecom = TreeDecompositionSolver().sample(bqm_diffusion)

toc = perf_counter()
print('random sampling: done ... ' + str(toc-tic) + ' sec.s')

print('first energy is: ' + str(sampleset_SA.first[1]))
print(len(sampleset_SA.lowest(rtol=SA_sample_rtol))/len(sampleset_SA))

np.save('SA_samples_' + sffx_1 + '_' + sffx_2 + '_' + sffx_3 + '.npy',
        sampleset_SA.record)
np.save('SA_samples_lowest_' + sffx_1 + '_' + sffx_2 + '_' + sffx_3 + '.npy',
        sampleset_SA.lowest(rtol=SA_sample_rtol).record)

# %% re-encode results: binary -> real
random_samp_bin_dict = sampleset_SA.first.sample
random_samp_real = np.zeros(n_real)
for ireal in range(0, n_real):
    for ibin in range(0, n_encode):
        bin_counter = ireal * n_encode + ibin
        random_samp_real[ireal] += 2**ibin * random_samp_bin_dict[bin_counter]
random_samp_real = random_samp_real / (2**n_encode - 1) * 2 - 1

# save
phi_h_random_samp = np.zeros(nen)
phi_h_random_samp[1:-1] = random_samp_real
npy_name = 'phi_h_' + sffx_1 + '_' + sffx_2 + '_' + sffx_3 + '.npy'
np.save(npy_name, phi_h_random_samp)

# %% visualization
u_exact = lambda x : np.sin(2.0 * np.pi * x)

solution = np.zeros(nen)
solution[1: -1] = np.load('classical_solution.npy')

x_ref = np.linspace(0, 1, 1001)
node_coord = np.linspace(0, 1, nen)

plt.figure(figsize=(4, 3))
plt.plot(x_ref, u_exact(x_ref), 'k-', label='Truth')
plt.plot(node_coord, phi_h_random_samp, 's--', markerfacecolor='none',
         label='SA')
plt.legend(loc='best')
plt.xlabel('Coordinates: $x$')
plt.ylabel('Solution')
plt.tight_layout()
plt.savefig('Simulated_Annealing.png', dpi=300)