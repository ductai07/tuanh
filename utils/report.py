import dcase_util
import os

class Reporter:
    def __init__(self, results_path, logger):
        self.results_path = results_path
        self.logger = logger
    
    def generate_report(self, metrics):
        """Tạo báo cáo tổng hợp"""
        # Tạo báo cáo tổng hợp với dcase_util
        report = dcase_util.utils.Example(
            title='Báo cáo phân tích âm thanh ESC-50',
            description='Kết quả phân tích và phân loại âm thanh với ESC-50 dataset',
            parameters={
                'Dataset': 'ESC-50',
                'Feature extraction': 'Mel Spectrogram',
                'Model': 'SB-CNN',
                'Folds': len(metrics),
                'Mean accuracy': f"{metrics['accuracy']:.4f}",
                'Mean F1-score': f"{metrics['f1_score']:.4f}",
                'Mean error rate': f"{metrics['error_rate']:.4f}"
            }
        )
        
        # In ra báo cáo
        self.logger.info(report)
        
        # Danh sách các hình ảnh kết quả
        result_images = [
            os.path.join(self.results_path, 'mel_dcase_util.png'),
            os.path.join(self.results_path, 'confusion_matrix_fold_1.png'),
            os.path.join(self.results_path, 'confusion_matrix_fold_2.png'),
            os.path.join(self.results_path, 'prediction_visualization.png'),
            os.path.join(self.results_path, 'top_classes.png')
        ]
        
        # Kiểm tra tất cả các hình ảnh có tồn tại không
        self.logger.info(f"Các hình ảnh kết quả đã được lưu trong thư mục: {self.results_path}")
        for img_path in result_images:
            if os.path.exists(img_path):
                self.logger.info(f"- {os.path.basename(img_path)}")
            else:
                self.logger.warning(f"Không tìm thấy: {os.path.basename(img_path)}")
        
        return report