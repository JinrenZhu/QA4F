# %% prelim
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import ExactSolver

# %% QUBO
Q = {('x', 'x'): -1, ('x', 'z'): 2, ('z', 'x'): 0, ('z', 'z'): -1}

# %% quantum sampler
# sampler = EmbeddingComposite(DWaveSampler())
# sampleset = sampler.sample_qubo(Q, num_reads=5000, label='LZ test - NOT Gate')

# %% quantum output
# print(sampleset)

# %% exact sampler
sampleset = ExactSolver.sample_qubo(Q=Q)
print(sampleset)