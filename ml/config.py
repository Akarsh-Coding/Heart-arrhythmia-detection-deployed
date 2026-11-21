# Signal & Window parameters

from pathlib import Path
import os


SAMPLING_RATE = 360  # Hz
WINDOW_SIZE = 360  # 180 samples pre-R-peak and 180 samples post-R-peak
NUM_CLASSES = 5  # Number of classes in the dataset

CLASSES = ['N', 'L', 'R', 'A', 'V']  # Literal MIT-BIH symbols used in the paper

CLASS_TO_INDEX = {name:i for i,name in enumerate(CLASSES)} 

# Record 102, 104, 107, 217 excluded per AAMI standard (paced beats)

DS1_TRAIN_RECORDS = [ '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230' ]

DS2_TEST_RECORDS = [ '100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234' ]

# Training Parameters
BATCH_SIZE = 64
CV_FOLDS = 2
PREPROCESSING = 'per_beat_zscore_v1'
EPOCHS = 30
INITIAL_LR = 0.001
L2_REG = 0.0001
DROPOUT_RATE = 0.5
RANDOM_SEED = 42

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'mitdb'
# Keep legacy artifacts separate: their labels/preprocessing are incompatible.
RUN_NAME = os.getenv('MSHAN_RUN', 'nlrav_v1')
MODEL_DIR = ROOT_DIR / 'models' / RUN_NAME
REPORTS_DIR = ROOT_DIR / 'reports' / RUN_NAME

# PhysioNet: records 201 and 202 are from the same subject.
# Preserve the published lists above for comparison; new fits omit 201.
TRAIN_RECORDS = [record for record in DS1_TRAIN_RECORDS if record != '201']
