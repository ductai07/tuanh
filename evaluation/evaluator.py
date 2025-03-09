import os
import json
import numpy as np
import matplotlib.pyplot as plt
import dcase_util
import sed_eval

class Evaluator:
    def __init__(self, dataset, results_path, logger):
        self.dataset = dataset
        self.results_path = results_path
        self.logger = logger
        self.fold_metrics = {}
        self.mean_metrics = {}
    
    def evaluate_fold(self, fold, y_true, y_pred, filenames):
        """Đánh giá kết quả của một fold"""
        # Sử dụng sed_eval để tính các metric đánh giá
        class_wise_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=self.dataset.get_classes()
        )
        
        # Chuyển đổi dự đoán sang định dạng event detections
        for i, (true_label, pred_label, filename) in enumerate(zip(y_true, y_pred, filenames)):
            true_label_name = self.dataset.get_classes()[true_label]
            pred_label_name = self.dataset.get_classes()[pred_label]
            
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
        
        self.logger.info(f"F1-score (macro): {eval_results['class_wise_average']['f_measure']['f_measure']:.4f}")
        self.logger.info(f"Error rate: {eval_results['class_wise_average']['error_rate']['error_rate']:.4f}")
        
        return eval_results
    
    def create_confusion_matrix(self, fold, y_pred, y_true):
        """Tạo và visualize ma trận nhầm lẫn"""
        cm = dcase_util.data.ProbabilityMatrix(
            data=[[1 if y_pred[i] == j else 0 for j in range(len(self.dataset.get_classes()))] for i in range(len(y_pred))],
            time_axis=0
        )
        
        # Visualize ma trận nhầm lẫn
        plt.figure(figsize=(12, 10))
        dcase_util.utils.plotting.plot_confusion_matrix(
            cm=np.sum(np.array(cm.data), axis=0),
            normalize=True,
            target_names=self.dataset.get_classes(),
            title=f'Ma trận nhầm lẫn cho fold {fold}'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f'confusion_matrix_fold_{fold}.png'))
        plt.close()
    
    def save_metrics(self):
        """Tính và lưu các metrics trung bình"""
        accuracies = [self.fold_metrics[f'fold_{fold}']['accuracy'] for fold in range(1, len(self.fold_metrics)+1)]
        f1_scores = [self.fold_metrics[f'fold_{fold}']['sed_eval']['class_wise_average']['f_measure']['f_measure'] for fold in range(1, len(self.fold_metrics)+1)]
        error_rates = [self.fold_metrics[f'fold_{fold}']['sed_eval']['class_wise_average']['error_rate']['error_rate'] for fold in range(1, len(self.fold_metrics)+1)]
        
        self.mean_metrics['accuracy'] = np.mean(accuracies)
        self.mean_metrics['f1_score'] = np.mean(f1_scores)
        self.mean_metrics['error_rate'] = np.mean(error_rates)
        
        # Lưu kết quả vào file JSON
        with open(os.path.join(self.results_path, 'evaluation_results.json'), 'w') as f:
            json.dump({
                'fold_metrics': {k: {kk: vv for kk, vv in v.items() if kk not in ['y_true', 'y_pred']} 
                               for k, v in self.fold_metrics.items()},
                'mean_metrics': self.mean_metrics
            }, f, indent=4)
        
        self.logger.info(f"Kết quả trung bình:")
        self.logger.info(f"Accuracy: {self.mean_metrics['accuracy']:.4f}")
        self.logger.info(f"F1-score: {self.mean_metrics['f1_score']:.4f}")
        self.logger.info(f"Error rate: {self.mean_metrics['error_rate']:.4f}")
        
        return self.mean_metrics