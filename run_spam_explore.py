import os
import subprocess
from datetime import datetime

if __name__ == "__main__":
    n_seeds = 30
    coefs = [0, 0.1, 0.3, 0.5, 0.7, 1, 2, 3, 4, 5]
    time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    parent_dir = "results/explore_exp/spam/"
    path = os.path.join(parent_dir, time)
    os.mkdir(path)

    for reg_coef in coefs:
        for seed in range(n_seeds):
            subprocess.check_call("./py-sbatch.sh %s %s %s %s" % ("spam_explore_exp.py", str(reg_coef), str(seed), path), shell=True)
