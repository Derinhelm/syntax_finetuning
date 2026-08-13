import json
from pathlib import Path


def collect_mean_metrics(root_output_dir_path):
    mean_dict = {}
    p = Path(root_output_dir_path)
    for m_path in p.glob(f"**/metrics_*.jsonl"):
        m_path_str = str(m_path)
        with open(m_path_str, 'r') as f:
            data = json.load(f)
        exp_name = m_path.stem
        mean_dict[exp_name] = data[f"{exp_name}_mean"]
    with open(f"{root_output_dir_path}/mean_metrics.json", 'w') as f:
        json.dump(mean_dict, f, indent=4)
