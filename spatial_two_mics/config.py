import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMIT_PATH = "/mnt/data/Speech/timit-wav"
DATASETS_DIR = "/mnt/nvme/spatial_two_mics_data/"
MODELS_DIR = "/mnt/nvme/spatial_two_mics_models/"
RESULTS_DIR = "/mnt/nvme/spatial_two_mics_results/"
MODELS_RAW_PHASE_DIR = "/mnt/nvme/spatial_two_mics_models_raw_phase/"
MODELS_GROUND_TRUTH = "/mnt/nvme/spatial_two_mics_models_ground_truth/"
FINAL_RESULTS_DIR = "/mnt/nvme/spatial_two_mics_final_eval_results/"

