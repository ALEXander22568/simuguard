"""Step by step comparison of the standalone program with SAPIEN's reports for one case, over the steps for which
contacts.txt holds SAPIEN's contacts (onset +- 400).  Prints the first steps where they differ.
usage: bowls_trace.py TAG [order] [contactDist]"""
import subprocess, sys, re, json
from pathlib import Path
import numpy as np
D = Path("/mnt/nvme0/shared/USER/simuguard/probe/physx_repro"); d = D / "data/bowls" / sys.argv[1]
order = sys.argv[2] if len(sys.argv) > 2 else "0"; cd = sys.argv[3] if len(sys.argv) > 3 else "0.02"
info = json.loads((d / "info.json").read_text()); o = info["onset"]
sap = {}
for l in open(d / "contacts.txt"):
    t = l.split(); n = int(t[2]); sap[int(t[0])] = (t[1], [float(x) for x in t[4:7]], sorted(float(x) for x in t[8:8 + n]))
rep = {}
p = subprocess.Popen([str(D / "build/repro"), str(d), order, "0.1", cd, "1", "1"], stdout=subprocess.PIPE, text=True, cwd=str(D))
for line in p.stdout:
    m = re.match(r"^step (\d+): ret \d+ contacts (\d+) manifoldPts \d+ n (\S+) (\S+) (\S+) sep(.*)$", line)
    if m: rep[int(m.group(1))] = ([float(m.group(i)) for i in (3, 4, 5)], sorted(float(x) for x in m.group(6).split()))
    m = re.match(r"^step (\d+): (bounds separated|pair in range)", line)
    if m and abs(int(m.group(1)) - o) < 3000: print(line.strip())
p.wait()
lo = min(sap) if sap else o - 400
bad = 0; same = 0
for s in range(max(lo, o - 400), o + 2):
    a, b = sap.get(s), rep.get(s)
    ok = (a is None and b is None) or (a is not None and b is not None and len(a[2]) == len(b[1]) and np.allclose(a[2], b[1], atol=2e-5) and np.allclose(a[1], b[0], atol=2e-4))
    if ok: same += 1; continue
    bad += 1
    if bad <= 12 or s >= o - 2:
        print("step", s, "| sapien", None if a is None else (a[0], np.round(a[1], 4).tolist(), np.round(np.array(a[2]) * 1e3, 2).tolist()), "| repro", None if b is None else (np.round(b[0], 4).tolist(), np.round(np.array(b[1]) * 1e3, 2).tolist()))
print("steps compared", same + bad, "identical", same, "different", bad, "first sapien step", lo)
