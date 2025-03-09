import os

# Định nghĩa đường dẫn
DATASET_PATH = 'ESC-50'
WORKSPACE = 'workspace'
FEATURES_PATH = os.path.join(WORKSPACE, 'features')
MODELS_PATH = os.path.join(WORKSPACE, 'models')
LOG_FOLDER = os.path.join(WORKSPACE, 'log')
RESULTS_PATH = os.path.join(WORKSPACE, 'results')

# Tạo các thư mục cần thiết
for directory in [WORKSPACE, FEATURES_PATH, MODELS_PATH, LOG_FOLDER, RESULTS_PATH]:
    os.makedirs(directory, exist_ok=True)

# Cấu hình tham số cho trích xuất đặc trưng
FEATURE_PARAMS = {
    'sr': 44100,
    'n_fft': 2048, 
    'hop_length': 1024,
    'n_mels': 64,
    'f_min': 0,
    'f_max': 22050
}

# Cấu hình tham số cho mô hình
MODEL_PARAMS = {
    'seed': 42,
    'batch_size': 64,
    'epochs': 30,
    'sequence_hop': 0.5,
    'sequence_length': 1.0,
    'preprocessing': 'normalization',
    'augmentation': None,
    'mixup': None
}