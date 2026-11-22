import json, sys
for path in sys.argv[1:]:
    d = json.load(open(path))
    print("REF", d["ref"])
    for g, pols in d["pooled"].items():
        for pol, t in pols.items():
            for c, v in t.items():
                n = v["n"]
                if not n:
                    continue
                sb, sa, sf, fs = v["success_before"], v["success_after"], v["flips_success_to_fail"], v["flips_fail_to_success"]
                print(f"  {g:22s} {pol:6s} {c:13s} n={n:2d} succ {sb[0]}/{n}->{sa[0]}/{n} {sa[2]} chg={v['success_change']:+.3f} s>f {sf[0]}/{sf[1]} {sf[2]} f>s {fs[0]}/{fs[1]} {fs[2]}")
