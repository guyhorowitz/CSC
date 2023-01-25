import os
import subprocess
from datetime import datetime

if __name__ == "__main__":
    n_seeds = 10
    n_dims = 5
    time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    parent_dir = "results/sensitivity_exp_v2/spam"
    path = os.path.join(parent_dir, time)
    os.mkdir(path)

    for n_flips in range(n_dims + 1):
        for seed in range(n_seeds):
            subprocess.check_call("./py-sbatch.sh %s %s %s %s" % ("spam_sensitivity_exp_v2.py", str(n_flips), str(seed), path), shell=True)
