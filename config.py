import numpy as np

BATCH_SIZE = 10
BATCHES_PER_EVALUATION = 100
DEBUG_LOG_ACTIVATIONS = False
DEBUG_LOG_DERIVATIVES = False
DEBUG_MODE = True
FLOAT_TYPE = np.float32
PROFILE = False

def start_profiler():
    if PROFILE:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

def stop_profiler():
    if PROFILE:
        pr.disable()
        pr.dump_stats("stats.prof")
