# This script requires Sympy package, which is not pre-installed in default Leap IDE

# %% prelim
import numpy as np
from sympy import symbols, simplify, latex, expand, diff, Array, collect
from scipy import interpolate
from time import perf_counter
import matplotlib.pyplot as plt

from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler

