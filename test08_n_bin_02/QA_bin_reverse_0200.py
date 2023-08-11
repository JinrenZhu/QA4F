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
n_encode = 2
n_bin = n_real * n_encode
range_encode = 20

# FEM
nel = 6
nen = nel * 2 + 1

# Sampler
QA_n_reads = 2000
QA_sample_rtol = 1.0e-4
QA_chain_strength = 500
# QA_time = 200

# suffix
sffx_1 = str(n_real).zfill(2)
sffx_2 = str(n_encode).zfill(2)
sffx_3 = str(QA_n_reads).zfill(5)
sffx_4 = str(QA_chain_strength).zfill(4)

# %% load QUBO coefficient
cov_matrix_off = np.load('cov_mat_off_' + sffx_1 + '_' + sffx_2 + '.npy')
bias_linear = np.load('bias_linear_' + sffx_1 + '_' + sffx_2 + '.npy')

# %% load results of Forward Annealing
QA_F_path = './'
QA_samples = np.load(QA_F_path + 'QA_samples_12_02_02000_0500_0200.npy')

QA_solution_vars = variables.Variables(range(0, n_bin))
QA_solution_bin = QA_samples['sample'][0]
print('load solution as follows')
print(QA_solution_bin)

QA_forward = dict(zip(QA_solution_vars, QA_solution_bin))

# %% sampling
# generate QUBO
bqm_channel = BinaryQuadraticModel('BINARY')
bqm_channel.add_linear_from_array(bias_linear)
bqm_channel.add_quadratic_from_dense(cov_matrix_off)

max_slope = 2.0

# test 01, energy=-78.02404252910627
# sffx_5 = str(1).zfill(2)
# reverse_schedule = make_reverse_anneal_schedule(s_target=0.6, hold_time=200, ramp_up_slope=max_slope)
# reinit_flag = False

# test 02, energy=-80.99657558377189
# sffx_5 = str(2).zfill(2)
# reverse_schedule = make_reverse_anneal_schedule(s_target=0.6, hold_time=200, ramp_up_slope=max_slope)
# reinit_flag = True

# test 03, energy=-80.99657558377189
sffx_5 = str(3).zfill(2)
reverse_schedule = make_reverse_anneal_schedule(s_target=0.5, hold_time=200, ramp_up_slope=max_slope)
reinit_flag = True

reverse_anneal_params = dict(anneal_schedule=reverse_schedule, initial_state=QA_forward, reinitialize_state=reinit_flag)

samplerQA = EmbeddingComposite(DWaveSampler())
sampleset_QA = samplerQA.sample(bqm_channel,
                                num_reads=QA_n_reads,
                                chain_strength=QA_chain_strength,
                                label='Channel-Reverse',
                                **reverse_anneal_params)

show(sampleset_QA) 

print(sampleset_QA.first)
print(sampleset_QA.record['sample'][0])
print(sampleset_QA.record['energy'][0])
print(len(sampleset_QA.lowest(rtol=QA_sample_rtol))/len(sampleset_QA))

sffx_all = sffx_1 + '_' + sffx_2 + '_' + sffx_3 + '_' + sffx_4 + '_' + sffx_5 + '.npy'
np.save('QA_samples_reverse_' + sffx_all, sampleset_QA.record)
# np.save('QA_samples_lowest_' + sffx_all, sampleset_QA.lowest(rtol=QA_sample_rtol).record)