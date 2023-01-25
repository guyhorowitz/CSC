import os
import subprocess
from datetime import datetime

import spam_params

if __name__ == "__main__":
    n_seeds = 15
    cost_scales = [20, 40, 80, 160]
    time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    parent_dir = "results/baselines_comparison/spam"
    path = os.path.join(parent_dir, time)
    os.mkdir(path)
    with open(f"{path}/global_params.txt", 'w') as f:
        for key, value in spam_params.params.items():
            f.write('%s: %s\n' % (key, value))

    for cost_scale in cost_scales:
        for seed in range(n_seeds):
            subprocess.check_call("./py-sbatch.sh %s %s %s %s" % ("spam_baselines_comparison_exp.py", str(cost_scale), str(seed), path), shell=True)
