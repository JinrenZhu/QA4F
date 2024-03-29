# Make sure the working path in the terminal is 'test04_time'.
# This script uses longer annealing time.
# %% prelim
import numpy as np

from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.inspector import show

# %% parameters
# 12 real varialbes * 10 binary encoding = 120 variables
n_real = 12
n_encode = 4
n_bin = n_real * n_encode
range_encode = 20
# FEM
nel = 6
nen = nel * 2 + 1
# Sampler
QA_n_reads = 2000
QA_sample_rtol = 1.0e-4
QA_chain_strength = 500
QA_time = 200
# suffix
sffx_1 = str(n_real).zfill(2)
sffx_2 = str(n_encode).zfill(2)
sffx_3 = str(QA_n_reads).zfill(5)
sffx_4 = str(QA_chain_strength).zfill(4)

# %% load QUBO coefficient
cov_matrix_off = np.load('cov_mat_off_' + sffx_1 + '_' + sffx_2 + '.npy')
bias_linear = np.load('bias_linear_' + sffx_1 + '_' + sffx_2 + '.npy')

# %% sampling
# generate QUBO
bqm_channel = BinaryQuadraticModel('BINARY')
bqm_channel.add_linear_from_array(bias_linear)
bqm_channel.add_quadratic_from_dense(cov_matrix_off)

# define anneal schedule
# 01 just try -100.9374596991726
# 0 ~ 20: slow evolving
# 20 ~ 22: quench
# quench_schedule=[[0.0, 0.0], [20.0, 0.5], [22.0, 1.0]]

# 02 steeper slope -102.01021573123536
# 0 ~ 20: slow evolving
# 20 ~ 20.5: quench
# quench_schedule=[[0.0, 0.0], [20.0, 0.5], [20.5, 1.0]]

# 03 longer time steeper slope -96.14760387870584
# 0 ~ 100: slow evolving 
# 100 ~ 100.5: quench:
# quench_schedule=[[0.0, 0.0], [100.0, 0.5], [100.5, 1.0]]

# 04 steeper slope: energy=-97.71853035655805
# 0 ~ 24: slow evolving
# 24 ~ 24.5: quench
# quench_schedule=[[0.0, 0.0], [24.0, 0.6], [24.5, 1.0]]

sffx_5 = str(4).zfill(2)
quench_schedule=[[0.0, 0.0], [24.0, 0.6], [24.5, 1.0]]

samplerQA = EmbeddingComposite(DWaveSampler())
sampleset_QA = samplerQA.sample(bqm_channel,
                                num_reads=QA_n_reads,
                                anneal_schedule=quench_schedule,
                                chain_strength=QA_chain_strength,
                                label='Channel-Quench')

show(sampleset_QA) 

print(sampleset_QA.first)
print(len(sampleset_QA.lowest(rtol=QA_sample_rtol))/len(sampleset_QA))

sffx_all = sffx_1 + '_' + sffx_2 + '_' + sffx_3 + '_' + sffx_4 + '_' + sffx_5 + '.npy'
np.save('QA_samples_quench' + sffx_all, sampleset_QA.record)
# np.save('QA_samples_lowest_' + sffx_all, sampleset_QA.lowest(rtol=QA_sample_rtol).record)