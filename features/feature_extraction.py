import os
import matplotlib.pyplot as plt
import dcase_util
from dcase_models.data.features import MelSpectrogram # type: ignore

class FeatureExtractor:
    def __init__(self, params, features_path, logger):
        self.params = params
        self.features_path = features_path
        self.logger = logger
        self.mel_features = MelSpectrogram(parameters=params)
    
    def extract(self, dataset):
        """Trích xuất đặc trưng cho toàn bộ dataset"""
        self.logger.info(f'Đang trích xuất đặc trưng Mel Spectrogram...')
        
        features_exist = os.path.exists(os.path.join(self.features_path, self.mel_features.get_name()))
        if not features_exist:
            self.mel_features.extract(dataset, self.features_path)
            self.logger.info('Đã hoàn thành trích xuất đặc trưng')
        else:
            self.logger.info('Đã tìm thấy đặc trưng đã trích xuất trước đó')
        
        return self.mel_features
    
    def visualize_features(self, file_path, results_path):
        """Visualize đặc trưng của một file âm thanh"""
        # Tải file âm thanh với dcase_util
        audio_container = dcase_util.containers.AudioContainer().load(file_path)
        
        # Trích xuất đặc trưng với dcase_util
        mel_extractor = dcase_util.features.MelExtractor(
            fs=self.params['sr'],
            win_length_samples=self.params['n_fft'],
            hop_length_samples=self.params['hop_length'],
            n_mels=self.params['n_mels'],
            fmin=self.params['f_min'],
            fmax=self.params['f_max']
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
        plt.savefig(os.path.join(results_path, 'mel_dcase_util.png'))
        plt.close()
        
        return mel_data