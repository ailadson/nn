import numpy as np

BATCH_SIZE = 10
BATCHES_PER_EVALUATION = 100
DEBUG_MODE = True
DEBUG_STEP_BY_STEP = DEBUG_MODE and False
DEBUG_LOG_ACTIVATIONS = DEBUG_STEP_BY_STEP and False
DEBUG_LOG_DERIVATIVES = DEBUG_STEP_BY_STEP and False
FLOAT_TYPE = np.float32
PROFILE = False

def float_zeros(size):
    return np.zeros(size, dtype=FLOAT_TYPE)

def start_profiler():
    if PROFILE:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

def stop_profiler():
    if PROFILE:
        pr.disable()
        pr.dump_stats("stats.prof")
