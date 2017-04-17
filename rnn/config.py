# dataset constants
DATASET_FILENAME = 'anna.txt'
READ_NEW_LINES = False
COLLAPSE_WHITESPACE = True
CASE_INSENSITIVE = True

# architecture constants
NUM_LAYERS = 2
NUM_LSTM_UNITS = 256
USE_MY_LSTM = False

# training constants
EPOCHS = 30
LEARNING_RATE = 0.001
CLIP_GRADIENT = 5
BATCH_SIZE = 10
STEP_SIZE = 100
PRINT_EPOCH_INFO_RATE = 100

# testing constants
TEST_MODEL_FILENAME = '__latest__'
SAMPLE_PRIME = "my love is"
SAMPLE_LENGTH = 158
LATEST_TEST_MODEL_FILENAME = '__latest__'

# TEST_VAR_NAME = "layers/0/basic_lstm_cell/weights:0"
