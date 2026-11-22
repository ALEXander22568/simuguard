import sys
import numpy as np
x, y = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in x.files:
    d = np.abs(x[k].astype(np.float64) - y[k].astype(np.float64))
    per_env = [bool((d[e] == 0).all()) for e in range(d.shape[0])]
    print(k, "equal", bool((d == 0).all()), "max", float(d.max()), "mean", round(float(d.mean()), 4), "frac_px_diff", round(float((d > 0).mean()), 4), "envs_equal", per_env)
