import numpy as np

BATCH_SIZE = 10
BATCHES_PER_EVALUATION = 10
DEBUG_MODE = False
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
