import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sed_eval
import dcase_util
import librosa
import warnings
warnings.filterwarnings('ignore')

# Cài đặt các thư viện (chạy trong terminal)
"""
# Cài đặt thư viện DCASE và các công cụ liên quan
pip install librosa
pip install kapre==0.3.7
pip install tensorflow==2.6.0
pip install h5py==3.1.0
pip install dcase-models
pip install sed_eval
pip install dcase_util
pip install sed_vis

# Clone repositories
git clone https://github.com/karolpiczak/ESC-50.git
"""

# Định nghĩa đường dẫn
DATASET_PATH = 'ESC-50'
WORKSPACE = 'workspace'
FEATURES_PATH = os.path.join(WORKSPACE, 'features')
MODELS_PATH = os.path.join(WORKSPACE, 'models')
LOG_FOLDER = os.path.join(WORKSPACE, 'log')
RESULTS_PATH = os.path.join(WORKSPACE, 'results')

# =============================================================================
# PHẦN 1: KHỞI TẠO VÀ CẤU HÌNH
# =============================================================================

# Tạo các thư mục cần thiết
for directory in [WORKSPACE, FEATURES_PATH, MODELS_PATH, LOG_FOLDER, RESULTS_PATH]:
    os.makedirs(directory, exist_ok=True)

# Khởi tạo logger sử dụng dcase_util
log = dcase_util.utils.FancyLogger()
log.title('Phân tích âm thanh tích hợp với ESC-50')
log.section_header('Khởi tạo')

# Import DCASE-models
from dcase_models.data.datasets import ESC50Dataset
from dcase_models.data.features import MelSpectrogram
from dcase_models.model.models import SB_CNN
from dcase_models.model.container import KerasModelContainer

# Khởi tạo dataset
log.info('Đang tải ESC-50 dataset...')
dataset = ESC50Dataset(DATASET_PATH)

# Thông tin dataset với dcase_util
dataset_info = dcase_util.datasets.dataset.Dataset(
    name='ESC-50',
    storage_path=DATASET_PATH,
    meta_filename='meta/esc50.csv'
)

# Hiển thị thông tin dataset
log.info(f'Tổng số mẫu: {dataset.get_total_samples()}')
log.info(f'Số lớp: {len(dataset.get_classes())}')
log.info(f'Các lớp: {dataset.get_classes()}')

# =============================================================================
# PHẦN 2: TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTION)
# =============================================================================
log.section_header('Trích xuất đặc trưng')

# Cấu hình tham số cho trích xuất đặc trưng
params_features = {
    'sr': 44100,
    'n_fft': 2048, 
    'hop_length': 1024,
    'n_mels': 64,
    'f_min': 0,
    'f_max': 22050
}

# Khởi tạo bộ trích xuất đặc trưng từ DCASE-models
mel_features = MelSpectrogram(parameters=params_features)
log.info(f'Đang trích xuất đặc trưng Mel Spectrogram...')

# Trích xuất đặc trưng (nếu chưa tồn tại)
features_exist = os.path.exists(os.path.join(FEATURES_PATH, mel_features.get_name()))
if not features_exist:
    mel_features.extract(dataset, FEATURES_PATH)
    log.info('Đã hoàn thành trích xuất đặc trưng')
else:
    log.info('Đã tìm thấy đặc trưng đã trích xuất trước đó')

# Sử dụng dcase_util để visualize đặc trưng
log.section_header('Visualization với dcase_util và sed_vis')

# Hàm visualize sử dụng dcase_util
def visualize_with_dcase_util(file_path):
    # Tải file âm thanh với dcase_util
    audio_container = dcase_util.containers.AudioContainer().load(file_path)
    
    # Trích xuất đặc trưng với dcase_util
    mel_extractor = dcase_util.features.MelExtractor(
        fs=params_features['sr'],
        win_length_samples=params_features['n_fft'],
        hop_length_samples=params_features['hop_length'],
        n_mels=params_features['n_mels'],
        fmin=params_features['f_min'],
        fmax=params_features['f_max']
    )
    
    mel_data = mel_extractor.extract(audio_container)
    
    # Visualize với dcase_util
    plt.figure(figsize=(10, 6))
    dcase_util.utils.plotting.plot_matrix(
        mel_data,
        title=f'Mel Spectrogram - {os.path.basename(file_path)}',
        x_axis='time',
        y_axis='mel_bands',
        cmap='inferno'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'mel_dcase_util.png'))
    plt.close()
    
    return mel_data

# Lấy một file mẫu
sample_file = dataset.get_files()[0]
log.info(f'Visualizing file mẫu: {os.path.basename(sample_file)}')
mel_data = visualize_with_dcase_util(sample_file)

# =============================================================================
# PHẦN 3: HUẤN LUYỆN MÔ HÌNH VỚI DCASE-MODELS
# =============================================================================
log.section_header('Huấn luyện mô hình')

# Cấu hình tham số cho mô hình
params_model = {
    'seed': 42,
    'batch_size': 64,
    'epochs': 30,  # Giảm số epochs để demo nhanh hơn
    'sequence_hop': 0.5,
    'sequence_length': 1.0,
    'preprocessing': 'normalization',
    'augmentation': None,
    'mixup': None
}

# Tạo một container mô hình với SB_CNN
model_container = KerasModelContainer(
    model=SB_CNN,
    model_path=MODELS_PATH,
    model_name='sb_cnn_esc50',
    metrics=['classification']
)

# Khởi tạo list để lưu kết quả của mỗi fold
fold_metrics = {}
mean_metrics = {}

# Thực hiện 2-fold cross-validation để demo (thông thường sẽ là 5-fold)
for fold in range(1, 3):
    log.info(f"Huấn luyện fold {fold}/2...")
    
    # Cấu hình dữ liệu huấn luyện và kiểm tra cho fold hiện tại
    data_train, data_val = dataset.get_fold(fold)
    
    # Huấn luyện mô hình
    model_container.train(
        data_train=data_train,
        data_val=data_val,
        features_path=FEATURES_PATH,
        features_name=mel_features.get_name(),
        sequences_params=params_model,
        params_learn=params_model
    )
    
    # Đánh giá mô hình trên tập validation
    metrics = model_container.evaluate(
        data_val,
        features_path=FEATURES_PATH,
        features_name=mel_features.get_name(),
        sequences_params=params_model
    )
    
    log.info(f"Fold {fold} - Accuracy: {metrics['accuracy']:.4f}")
    fold_metrics[f'fold_{fold}'] = metrics
    
    # Dự đoán trên tập validation để tạo ma trận nhầm lẫn
    y_pred = model_container.predict(
        data_val,
        features_path=FEATURES_PATH,
        features_name=mel_features.get_name(),
        sequences_params=params_model
    )
    
    # Lấy ground truth
    y_true = []
    filenames = []
    for file_names in data_val['file_names']:
        for file_name in file_names:
            y_true.append(dataset.get_annotations(file_name))
            filenames.append(file_name)
    
    # Lưu dự đoán và ground truth để đánh giá
    fold_metrics[f'fold_{fold}']['y_true'] = y_true
    fold_metrics[f'fold_{fold}']['y_pred'] = y_pred.tolist()
    
    # Sử dụng sed_eval để tính các metric đánh giá
    class_wise_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=dataset.get_classes()
    )
    
    # Chuyển đổi dự đoán sang định dạng event detections
    for i, (true_label, pred_label, filename) in enumerate(zip(y_true, y_pred, filenames)):
        true_label_name = dataset.get_classes()[true_label]
        pred_label_name = dataset.get_classes()[pred_label]
        
        ref_event = [
            {
                'event_label': true_label_name,
                'event_onset': 0.0,
                'event_offset': 5.0,  # ESC-50 files are 5 seconds long
                'file': filename
            }
        ]
        
        est_event = [
            {
                'event_label': pred_label_name,
                'event_onset': 0.0,
                'event_offset': 5.0,
                'file': filename
            }
        ]
        
        class_wise_metrics.evaluate(
            reference_event_list=ref_event,
            estimated_event_list=est_event
        )
    
    # Lấy kết quả từ sed_eval
    eval_results = class_wise_metrics.results()
    log.info(f"F1-score (macro): {eval_results['class_wise_average']['f_measure']['f_measure']:.4f}")
    log.info(f"Error rate: {eval_results['class_wise_average']['error_rate']['error_rate']:.4f}")
    
    # Lưu kết quả đánh giá
    fold_metrics[f'fold_{fold}']['sed_eval'] = eval_results
    
    # Tạo ma trận nhầm lẫn với dcase_util
    cm = dcase_util.data.ProbabilityMatrix(
        data=[[1 if y_pred[i] == j else 0 for j in range(len(dataset.get_classes()))] for i in range(len(y_pred))],
        time_axis=0
    )
    
    # Visualize ma trận nhầm lẫn
    plt.figure(figsize=(12, 10))
    dcase_util.utils.plotting.plot_confusion_matrix(
        cm=np.sum(np.array(cm.data), axis=0),
        normalize=True,
        target_names=dataset.get_classes(),
        title=f'Ma trận nhầm lẫn cho fold {fold}'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'confusion_matrix_fold_{fold}.png'))
    plt.close()

# Tính trung bình các metric
accuracies = [fold_metrics[f'fold_{fold}']['accuracy'] for fold in range(1, 3)]
f1_scores = [fold_metrics[f'fold_{fold}']['sed_eval']['class_wise_average']['f_measure']['f_measure'] for fold in range(1, 3)]
error_rates = [fold_metrics[f'fold_{fold}']['sed_eval']['class_wise_average']['error_rate']['error_rate'] for fold in range(1, 3)]

mean_metrics['accuracy'] = np.mean(accuracies)
mean_metrics['f1_score'] = np.mean(f1_scores)
mean_metrics['error_rate'] = np.mean(error_rates)

# Lưu kết quả vào file JSON
with open(os.path.join(RESULTS_PATH, 'evaluation_results.json'), 'w') as f:
    json.dump({
        'fold_metrics': {k: v for k, v in fold_metrics.items() if k not in ['y_true', 'y_pred']},
        'mean_metrics': mean_metrics
    }, f, indent=4)

log.info(f"Kết quả trung bình:")
log.info(f"Accuracy: {mean_metrics['accuracy']:.4f}")
log.info(f"F1-score: {mean_metrics['f1_score']:.4f}")
log.info(f"Error rate: {mean_metrics['error_rate']:.4f}")

# =============================================================================
# PHẦN 4: DEMO DỰ ĐOÁN VÀ VISUALIZATION
# =============================================================================
log.section_header('Demo dự đoán')

# Hàm dự đoán với SED visualization
def predict_and_visualize(file_path):
    # Trích xuất đặc trưng
    mel_features.extract_file(file_path, FEATURES_PATH)
    
    # Dự đoán
    y_pred = model_container.predict_file(
        file_path,
        features_path=FEATURES_PATH,
        features_name=mel_features.get_name(),
        sequences_params=params_model
    )
    
    # Tải file âm thanh
    audio_container = dcase_util.containers.AudioContainer().load(file_path)
    
    # Trích xuất đặc trưng
    mel_data = mel_features.load_feature(file_path, FEATURES_PATH)
    
    # Tạo predicted events
    class_name = dataset.get_classes()[y_pred[0]]
    event_label = {
        'event_label': class_name,
        'event_onset': 0.0,
        'event_offset': audio_container.duration_sec,
        'probability': 1.0
    }
    
    # Visualize kết quả dự đoán
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.title(f'Waveform - Predicted Class: {class_name}')
    plt.plot(np.arange(len(audio_container.data)) / audio_container.fs, audio_container.data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    dcase_util.utils.plotting.plot_matrix(
        mel_data,
        title='Mel Spectrogram',
        x_axis='time',
        y_axis='mel_bands',
        cmap='inferno'
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'prediction_visualization.png'))
    plt.close()
    
    # Lấy top 3 lớp với xác suất cao nhất
    class_probs = model_container.predict_file_proba(
        file_path,
        features_path=FEATURES_PATH,
        features_name=mel_features.get_name(),
        sequences_params=params_model
    )[0]
    
    top_indices = np.argsort(class_probs)[-3:][::-1]
    top_classes = [dataset.get_classes()[i] for i in top_indices]
    top_probs = [class_probs[i] for i in top_indices]
    
    # Visualize top classes
    plt.figure(figsize=(8, 5))
    plt.bar(top_classes, top_probs)
    plt.title('Top 3 lớp có xác suất cao nhất')
    plt.xlabel('Lớp')
    plt.ylabel('Xác suất')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'top_classes.png'))
    plt.close()
    
    # In kết quả
    log.info(f"Dự đoán lớp: {class_name}")
    log.info("Top 3 lớp có xác suất cao nhất:")
    for cls, prob in zip(top_classes, top_probs):
        log.info(f"{cls}: {prob:.4f}")
    
    return y_pred, class_probs

# Chọn một file mẫu để dự đoán
sample_file = dataset.get_files()[10]  # Chọn một file khác với file trước đó
log.info(f"Demo dự đoán trên file: {os.path.basename(sample_file)}")
y_pred, class_probs = predict_and_visualize(sample_file)

# =============================================================================
# PHẦN 5: ĐÁNH GIÁ THỊ GIÁC VỚI SED_VIS
# =============================================================================
# Lưu ý: sed_vis yêu cầu cài đặt bổ sung và có thể không tương thích với môi trường hiện tại
# Đoạn mã sau đây là minh họa cách sử dụng, có thể cần điều chỉnh tùy thuộc vào phiên bản

try:
    import sed_vis
    
    log.section_header('Visualization với sed_vis')
    
    # Tạo event rolls cho sed_vis
    all_events = []
    
    # Lấy một số file mẫu từ dataset
    sample_files = dataset.get_files()[:5]
    
    for file_path in sample_files:
        # Dự đoán
        y_pred = model_container.predict_file(
            file_path,
            features_path=FEATURES_PATH,
            features_name=mel_features.get_name(),
            sequences_params=params_model
        )[0]
        
        class_name = dataset.get_classes()[y_pred]
        
        # Tạo event
        event = {
            'event_label': class_name,
            'event_onset': 0.0,
            'event_offset': 5.0,  # ESC-50 files are 5 seconds long
            'file': file_path
        }
        
        all_events.append(event)
    
    # Lưu events dưới dạng JSON
    with open(os.path.join(RESULTS_PATH, 'predicted_events.json'), 'w') as f:
        json.dump(all_events, f, indent=4)
    
    log.info("Đã lưu predicted events vào file JSON")
    
    # Lưu ý về cách sử dụng sed_vis (thông thường cần giao diện đồ họa)
    log.info("Để visualize với sed_vis, sử dụng API của nó:")
    log.info("1. Tải audio_container với dcase_util")
    log.info("2. Tạo event lists với định dạng sed_eval")
    log.info("3. Sử dụng sed_vis.visualization.event_roll()")
    log.info("4. Lưu kết quả hoặc hiển thị trong notebook")

except ImportError:
    log.info("sed_vis không được cài đặt hoặc không tương thích. Bỏ qua visualization với sed_vis.")

# =============================================================================
# PHẦN 6: TẠO BÁO CÁO TỔNG HỢP
# =============================================================================
log.section_header('Tạo báo cáo tổng hợp')

# Tạo báo cáo tổng hợp với dcase_util
report = dcase_util.utils.Example(
    title='Báo cáo phân tích âm thanh ESC-50',
    description='Kết quả phân tích và phân loại âm thanh với ESC-50 dataset',
    parameters={
        'Dataset': 'ESC-50',
        'Feature extraction': 'Mel Spectrogram',
        'Model': 'SB-CNN',
        'Folds': 2,
        'Mean accuracy': f"{mean_metrics['accuracy']:.4f}",
        'Mean F1-score': f"{mean_metrics['f1_score']:.4f}",
        'Mean error rate': f"{mean_metrics['error_rate']:.4f}"
    }
)

# In ra báo cáo
log.info(report)

# Danh sách các hình ảnh kết quả
result_images = [
    os.path.join(RESULTS_PATH, 'mel_dcase_util.png'),
    os.path.join(RESULTS_PATH, 'confusion_matrix_fold_1.png'),
    os.path.join(RESULTS_PATH, 'confusion_matrix_fold_2.png'),
    os.path.join(RESULTS_PATH, 'prediction_visualization.png'),
    os.path.join(RESULTS_PATH, 'top_classes.png')
]

# Kiểm tra tất cả các hình ảnh có tồn tại không
log.info(f"Các hình ảnh kết quả đã được lưu trong thư mục: {RESULTS_PATH}")
for img_path in result_images:
    if os.path.exists(img_path):
        log.info(f"- {os.path.basename(img_path)}")
    else:
        log.warning(f"Không tìm thấy: {os.path.basename(img_path)}")

log.section_header('Kết luận')
log.info('Đã hoàn thành phân tích âm thanh tích hợp với ESC-50')
log.info(f'Kết quả đã được lưu trong thư mục: {WORKSPACE}')
log.info('Sử dụng các kết quả để so sánh các phương pháp trích xuất đặc trưng và mô hình khác nhau')
