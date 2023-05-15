# All constants are saved in this file
import numpy as np

THRESHOLDS = [
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.035,
    0.04,
    0.045,
    0.05,
    0.055,
    0.06,
    0.065,
    0.07,
    0.075,
    0.08,
    0.085,
    0.09,
    0.095,
    0.10,
    0.125,
    0.15,
    0.20,
    0.3,
    0.5,
]
THRESHOLDS_LOG = np.log([x + 1 for x in THRESHOLDS])
CVAR_LEVEL = [0.1, 0.5, 1, 2.5, 5, 10]
CI_LEVELS = [0.95]
NUM_BOOTSTRAPPING = 10
STOP_LOSS = 0.075
BOOL_TO_EXCEL = True
times = []