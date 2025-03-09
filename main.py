import os
import warnings
warnings.filterwarnings('ignore')

# Import các module
from config.config import *
from utils.logger import Logger
from data.dataset import DatasetHandler
from features.feature_extraction import FeatureExtractor
from models.model import ModelHandler
from evaluation.evaluator import Evaluator
from visualization.visualize import Visualizer
from utils.report import Reporter

def main():
    # Khởi tạo logger
    logger = Logger("Phân tích âm thanh tích hợp với ESC-50")
    logger.section("Khởi tạo")
    
    # Tạo dataset handler
    dataset_handler = DatasetHandler(DATASET_PATH, logger)
    dataset = dataset_handler.load()
    
    # Trích xuất đặc trưng
    logger.section("Trích xuất đặc trưng")
    feature_extractor = FeatureExtractor(FEATURE_PARAMS, FEATURES_PATH, logger)
    mel_features = feature_extractor.extract(dataset)
    
    # Visualize một file mẫu
    sample_file = dataset_handler.get_sample(0)
    logger.info(f'Visualizing file mẫu: {os.path.basename(sample_file)}')
    mel_data = feature_extractor.visualize_features(sample_file, RESULTS_PATH)
    
    # Khởi tạo model
    logger.section("Huấn luyện mô hình")
    model_handler = ModelHandler(MODELS_PATH, logger)
    model_container = model_handler.create_model()
    
    # Khởi tạo evaluator
    evaluator = Evaluator(dataset, RESULTS_PATH, logger)
    
    # Huấn luyện và đánh giá trên các fold
    for fold in range(1, 3):  # 2-fold cross-validation
        logger.info(f"Huấn luyện fold {fold}/2...")
        
        # Cấu hình dữ liệu huấn luyện và kiểm tra cho fold hiện tại
        data_train, data_val = dataset_handler.get_fold(fold)
        
        # Huấn luyện mô hình
        model_handler.train(data_train, data_val, FEATURES_PATH, mel_features.get_name(), MODEL_PARAMS)
        
        # Đánh giá mô hình
        metrics = model_handler.evaluate(data_val, FEATURES_PATH, mel_features.get_name(), MODEL_PARAMS)
        logger.info(f"Fold {fold} - Accuracy: {metrics['accuracy']:.4f}")
        
        # Dự đoán để tạo ma trận nhầm lẫn
        y_pred = model_handler.predict(data_val, FEATURES_PATH, mel_features.get_name(), MODEL_PARAMS)
        
        # Lấy ground truth
        y_true = []
        filenames = []
        for file_names in data_val['file_names']:
            for file_name in file_names:
                y_true.append(dataset.get_annotations(file_name))
                filenames.append(file_name)
        
        # Lưu vào fold_metrics
        evaluator.fold_metrics[f'fold_{fold}'] = metrics
        evaluator.fold_metrics[f'fold_{fold}']['y_true'] = y_true
        evaluator.fold_metrics[f'fold_{fold}']['y_pred'] = y_pred.tolist()
        
        # Đánh giá với sed_eval
        eval_results = evaluator.evaluate_fold(fold, y_true, y_pred, filenames)
        evaluator.fold_metrics[f'fold_{fold}']['sed_eval'] = eval_results
        
        # Tạo ma trận nhầm lẫn
        evaluator.create_confusion_matrix(fold, y_pred, y_true)
    
    # Lưu metrics
    mean_metrics = evaluator.save_metrics()
    
    # Demo dự đoán
    logger.section("Demo dự đoán")
    visualizer = Visualizer(dataset, RESULTS_PATH, logger)
    
    # Chọn một file mẫu để dự đoán
    sample_file = dataset_handler.get_sample(10)
    logger.info(f"Demo dự đoán trên file: {os.path.basename(sample_file)}")
    
    # Trích xuất đặc trưng cho file demo
    mel_features.extract_file(sample_file, FEATURES_PATH)
    
    # Dự đoán
    y_pred = model_handler.predict_file(sample_file, FEATURES_PATH, mel_features.get_name(), MODEL_PARAMS)
    class_probs = model_handler.predict_file_proba(sample_file, FEATURES_PATH, mel_features.get_name(), MODEL_PARAMS)
    
    # Visualize kết quả dự đoán
    mel_data = mel_features.load_feature(sample_file, FEATURES_PATH)
    visualizer.visualize_prediction(sample_file, y_pred, mel_data, class_probs)
    
    # Tạo báo cáo tổng hợp
    logger.section("Tạo báo cáo tổng hợp")
    reporter = Reporter(RESULTS_PATH, logger)
    reporter.generate_report(mean_metrics)
    
    # Kết luận
    logger.section("Kết luận")
    logger.info('Đã hoàn thành phân tích âm thanh tích hợp với ESC-50')
    logger.info(f'Kết quả đã được lưu trong thư mục: {WORKSPACE}')
    logger.info('Sử dụng các kết quả để so sánh các phương pháp trích xuất đặc trưng và mô hình khác nhau')

if __name__ == "__main__":
    main()