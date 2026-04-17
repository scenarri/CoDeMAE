import os
import json
import glob


def get_formatted_results(model_name):
    datasets = ['PIE', 'DDHR-SK', 'WHU', 'DFC20', 'BEN', 'EuroSat']
    seeds = ['0', '1', '2', '3', '4']
    
    dataset_averages = []
    
    for dataset in datasets:
        for modality in ['RGB', 'SAR']:
            dataset_metrics = []
            for seed in seeds:
                pattern = f"./output/{model_name}-{dataset}-{modality}*-seed-{seed}_log.json"
                print(pattern)
                files = glob.glob(pattern)
                
                if files:
                    file_path = files[0]
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if "best_metric" in data:
                                dataset_metrics.append(float(data["best_metric"]))
                    except Exception as e:
                        print(f"error at loading {file_path}: {e}")
            
            # average across seeds
            if dataset_metrics:
                avg_score = sum(dataset_metrics) / len(dataset_metrics)
                dataset_averages.append(f"{avg_score:.2f}")
            else:
                dataset_averages.append("N/A")
    
    return " & ".join(dataset_averages)

if __name__ == '__main__':
    s = get_formatted_results('DOFA')
    print(s)