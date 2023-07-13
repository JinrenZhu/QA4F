# Make sure the working path in the terminal is 'test03'.

# %% prelim
import numpy as np

from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite

# %% parameters
# 12 real varialbes * 10 binary encoding = 120 variables
n_real = 12
n_encode = 6
n_bin = n_real * n_encode
range_encode = 20
# FEM
nel = 6
nen = nel * 2 + 1
# Sampler
QA_n_reads = 2000
QA_sample_rtol = 1.0e-4
# suffix
sffx_1 = str(n_real).zfill(2)
sffx_2 = str(n_encode).zfill(2)
sffx_3 = str(QA_n_reads).zfill(5)

# %% load QUBO coefficient
cov_matrix_off = np.load('cov_mat_off_' + sffx_1 + '_' + sffx_2 + '.npy')
bias_linear = np.load('bias_linear_' + sffx_1 + '_' + sffx_2 + '.npy')

# %% sampling
# generate QUBO
bqm_channel = BinaryQuadraticModel('BINARY')
bqm_channel.add_linear_from_array(bias_linear)
bqm_channel.add_quadratic_from_dense(cov_matrix_off)

samplerQA = EmbeddingComposite(DWaveSampler())
sampleset_QA = samplerQA.sample(bqm_channel, num_reads=QA_n_reads, label='QFI-Channel-6bits')

print(sampleset_QA.first)
print(len(sampleset_QA.lowest(rtol=QA_sample_rtol))/len(sampleset_QA))

np.save('QA_samples_' + sffx_1 + '_' + sffx_2 + '_' + sffx_3 + '.npy',
        sampleset_QA.record)
np.save('QA_samples_lowest' + sffx_1 + '_' + sffx_2 + '_' + sffx_3 + '.npy',
        sampleset_QA.lowest(rtol=QA_sample_rtol).record)