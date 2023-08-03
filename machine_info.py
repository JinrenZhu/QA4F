# This script reads some parameters of the machines

# %% prelim
import numpy as np

from dwave.system.samplers import DWaveSampler

# %% machine info
sampler = DWaveSampler()   

# name
print("Connected to sampler", sampler.solver.name)
print("Maximum anneal-schedule points: {}".format(sampler.properties["max_anneal_schedule_points"]))
print("Annealing time range: {}".format(sampler.properties["annealing_time_range"]))
print("Default annealing time: {}".format(sampler.properties["default_annealing_time"]))

max_slope = 1.0/sampler.properties["annealing_time_range"][0]
print("Maximum slope allowed on this solver is {:.2f}.".format(max_slope))