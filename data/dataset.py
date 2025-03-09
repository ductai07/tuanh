from dcase_models.data.datasets import ESC50Dataset  
import dcase_util

class DatasetHandler:
    def __init__(self, dataset_path, logger):
        self.logger = logger
        self.dataset_path = dataset_path
        self.dataset = None
        self.dataset_info = None
        
    def load(self):
        """Tải dataset"""
        self.logger.info('Đang tải ESC-50 dataset...')
        self.dataset = ESC50Dataset(self.dataset_path)
        
        # Thông tin dataset với dcase_util
        self.dataset_info = dcase_util.datasets.dataset.Dataset(
            name='ESC-50',
            storage_path=self.dataset_path,
            meta_filename='meta/esc50.csv'
        )
        
        # Hiển thị thông tin dataset
        self.logger.info(f'Tổng số mẫu: {self.dataset.get_total_samples()}')
        self.logger.info(f'Số lớp: {len(self.dataset.get_classes())}')
        self.logger.info(f'Các lớp: {self.dataset.get_classes()}')
        
        return self.dataset
    
    def get_fold(self, fold):
        """Lấy dữ liệu cho fold cụ thể"""
        return self.dataset.get_fold(fold)
    
    def get_sample(self, index=0):
        """Lấy một file mẫu"""
        return self.dataset.get_files()[index]
    
    def get_classes(self):
        """Lấy danh sách các lớp"""
        return self.dataset.get_classes()