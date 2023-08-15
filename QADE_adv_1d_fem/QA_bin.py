# Make sure the working path in the terminal is 'test04_time'.
# This script uses longer annealing time.
# %% prelim
import numpy as np

from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.inspector import show

# %% parameters
# FEM
nel = 8
nen = nel + 1
# encode
n_real = nen - 2
n_encode = 4
n_bin = n_real * n_encode
# Sampler
QA_n_reads = 2000
QA_sample_rtol = 1.0e-4
QA_chain_strength = 100
QA_time = 20
# suffix
sffx_1 = str(n_real).zfill(2)
sffx_2 = str(n_encode).zfill(2)
sffx_3 = str(QA_n_reads).zfill(5)
sffx_4 = str(QA_chain_strength).zfill(4)
sffx_5 = str(QA_time).zfill(4)

# %% load QUBO coefficient
cov_matrix_off = np.load('cov_mat_off_' + sffx_1 + '_' + sffx_2 + '.npy')
bias_linear = np.load('bias_linear_' + sffx_1 + '_' + sffx_2 + '.npy')

# %% sampling
# generate QUBO
bqm_channel = BinaryQuadraticModel('BINARY')
bqm_channel.add_linear_from_array(bias_linear)
bqm_channel.add_quadratic_from_dense(cov_matrix_off)

samplerQA = EmbeddingComposite(DWaveSampler())
sampleset_QA = samplerQA.sample(bqm_channel,
                                num_reads=QA_n_reads,
                                annealing_time=QA_time,
                                chain_strength=QA_chain_strength,
                                label='Adv-FEM-4bits')

show(sampleset_QA) 

print(sampleset_QA.first)
print(len(sampleset_QA.lowest(rtol=QA_sample_rtol))/len(sampleset_QA))

sffx_all = sffx_1 + '_' + sffx_2 + '_' + sffx_3 + '_' + sffx_4 + '_' + sffx_5 + '.npy'
np.save('QA_samples_' + sffx_all, sampleset_QA.record)
# np.save('QA_samples_lowest_' + sffx_all, sampleset_QA.lowest(rtol=QA_sample_rtol).record)