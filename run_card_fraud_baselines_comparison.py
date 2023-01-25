import os
import subprocess
from datetime import datetime

import card_fraud_params

if __name__ == "__main__":
    n_seeds = 15
    cost_scales = [0.5, 1, 2, 4]
    time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    parent_dir = "results/baselines_comparison/card_fraud"
    path = os.path.join(parent_dir, time)
    os.mkdir(path)
    with open(f"{path}/global_params.txt", 'w') as f:
        for key, value in card_fraud_params.params.items():
            f.write('%s: %s\n' % (key, value))

    for cost_scale in cost_scales:
        for seed in range(n_seeds):
            subprocess.check_call("./py-sbatch.sh %s %s %s %s" % ("card_fraud_baselines_comparison_exp.py", str(cost_scale), str(seed), path), shell=True)
