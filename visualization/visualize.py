import os
import numpy as np
import matplotlib.pyplot as plt
import dcase_util
import json

class Visualizer:
    def __init__(self, dataset, results_path, logger):
        self.dataset = dataset
        self.results_path = results_path
        self.logger = logger
    
    def visualize_prediction(self, file_path, y_pred, mel_data, class_probs):
        """Visualize dự đoán cho một file audio"""
        # Tải file âm thanh
        audio_container = dcase_util.containers.AudioContainer().load(file_path)
        
        # Lấy class name từ dự đoán
        class_name = self.dataset.get_classes()[y_pred[0]]
        
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
        plt.savefig(os.path.join(self.results_path, 'prediction_visualization.png'))
        plt.close()
        
        # Lấy top 3 lớp với xác suất cao nhất
        top_indices = np.argsort(class_probs[0])[-3:][::-1]
        top_classes = [self.dataset.get_classes()[i] for i in top_indices]
        top_probs = [class_probs[0][i] for i in top_indices]
        
        # Visualize top classes
        plt.figure(figsize=(8, 5))
        plt.bar(top_classes, top_probs)
        plt.title('Top 3 lớp có xác suất cao nhất')
        plt.xlabel('Lớp')
        plt.ylabel('Xác suất')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'top_classes.png'))
        plt.close()
        
        # In kết quả
        self.logger.info(f"Dự đoán lớp: {class_name}")
        self.logger.info("Top 3 lớp có xác suất cao nhất:")
        for cls, prob in zip(top_classes, top_probs):
            self.logger.info(f"{cls}: {prob:.4f}")
        
    def save_predicted_events(self, files, predictions):
        """Lưu dự đoán dưới dạng events"""
        all_events = []
        
        for file_path, y_pred in zip(files, predictions):
            class_name = self.dataset.get_classes()[y_pred]
            
            # Tạo event
            event = {
                'event_label': class_name,
                'event_onset': 0.0,
                'event_offset': 5.0,  # ESC-50 files are 5 seconds long
                'file': file_path
            }
            
            all_events.append(event)
        
        # Lưu events dưới dạng JSON
        with open(os.path.join(self.results_path, 'predicted_events.json'), 'w') as f:
            json.dump(all_events, f, indent=4)
        
        self.logger.info("Đã lưu predicted events vào file JSON")