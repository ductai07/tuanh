from dcase_models.model.models import SB_CNN  
from dcase_models.model.container import KerasModelContainer  

class ModelHandler:
    def __init__(self, models_path, logger):
        self.models_path = models_path
        self.logger = logger
        self.model_container = None
    
    def create_model(self):
        """Tạo container mô hình SB_CNN"""
        self.model_container = KerasModelContainer(
            model=SB_CNN,
            model_path=self.models_path,
            model_name='sb_cnn_esc50',
            metrics=['classification']
        )
        return self.model_container
    
    def train(self, data_train, data_val, features_path, features_name, params):
        """Huấn luyện mô hình"""
        self.logger.info(f"Bắt đầu huấn luyện mô hình...")
        self.model_container.train(
            data_train=data_train,
            data_val=data_val,
            features_path=features_path,
            features_name=features_name,
            sequences_params=params,
            params_learn=params
        )
    
    def evaluate(self, data_val, features_path, features_name, params):
        """Đánh giá mô hình"""
        metrics = self.model_container.evaluate(
            data_val,
            features_path=features_path,
            features_name=features_name,
            sequences_params=params
        )
        return metrics
    
    def predict(self, data, features_path, features_name, params):
        """Dự đoán trên tập dữ liệu"""
        y_pred = self.model_container.predict(
            data,
            features_path=features_path,
            features_name=features_name,
            sequences_params=params
        )
        return y_pred
    
    def predict_file(self, file_path, features_path, features_name, params):
        """Dự đoán cho một file"""
        return self.model_container.predict_file(
            file_path,
            features_path=features_path,
            features_name=features_name,
            sequences_params=params
        )
    
    def predict_file_proba(self, file_path, features_path, features_name, params):
        """Dự đoán xác suất cho một file"""
        return self.model_container.predict_file_proba(
            file_path,
            features_path=features_path,
            features_name=features_name,
            sequences_params=params
        )