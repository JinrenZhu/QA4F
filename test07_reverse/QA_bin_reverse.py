# Make sure the working path in the terminal is 'test04_time'.
# This script uses longer annealing time.
# %% prelim
import numpy as np

from dimod import BinaryQuadraticModel
from dimod import variables
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.inspector import show
from schedule import make_reverse_anneal_schedule

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

# %% load results of Forward Annealing
QA_F_path = '../test03/'
QA_samples = np.load(QA_F_path + 'QA_samples_lowest_12_04_02000_0500.npy')

QA_solution_vars = variables.Variables(range(0, n_bin))
QA_solution_bin = QA_samples['sample'][0]

QA_forward = dict(zip(QA_solution_vars, QA_solution_bin))

# %% sampling
# generate QUBO
bqm_channel = BinaryQuadraticModel('BINARY')
bqm_channel.add_linear_from_array(bias_linear)
bqm_channel.add_quadratic_from_dense(cov_matrix_off)



sffx_5 = str(2).zfill(2)
pause_schedule=[[0.0, 0.0], [8.0, 0.4], [108.0, 0.4], [120.0, 1.0]]

samplerQA = EmbeddingComposite(DWaveSampler())
sampleset_QA = samplerQA.sample(bqm_channel,
                                num_reads=QA_n_reads,
                                anneal_schedule=pause_schedule,
                                chain_strength=QA_chain_strength,
                                label='Channel-Pause')

show(sampleset_QA) 

print(sampleset_QA.first)
print(len(sampleset_QA.lowest(rtol=QA_sample_rtol))/len(sampleset_QA))

sffx_all = sffx_1 + '_' + sffx_2 + '_' + sffx_3 + '_' + sffx_4 + '_' + sffx_5 + '.npy'
np.save('QA_samples_pause' + sffx_all, sampleset_QA.record)
# np.save('QA_samples_lowest_' + sffx_all, sampleset_QA.lowest(rtol=QA_sample_rtol).record)