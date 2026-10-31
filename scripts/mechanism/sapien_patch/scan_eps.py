"""Find every instruction in libsapien.so that references a constant equal to FLT_EPSILON (0x34000000),
group by function symbol, and print the context of the scalar compares (comiss/ucomiss)."""
import subprocess, struct, re, sys, bisect, json
L = sys.argv[1]; out_json = sys.argv[2] if len(sys.argv) > 2 else None
data = open(L, "rb").read()
# sections
secs = {}
for line in subprocess.run(["readelf", "-S", "-W", L], capture_output=True, text=True).stdout.splitlines():
    m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
    if m: secs[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16), int(m.group(4), 16))
ro = [s for s in secs if s.startswith(".rodata")]
eps_addrs = set()
for s in ro:
    va, off, size = secs[s]
    for i in range(0, size - 3, 4):
        if data[off + i: off + i + 4] == b"\x00\x00\x00\x34": eps_addrs.add(va + i)
print("rodata sections", ro, "FLT_EPSILON constants at", len(eps_addrs), "addresses")
new = [va + i for s in ro for va, off, size in [secs[s]] for i in range(0, size - 3, 4) if data[off + i: off + i + 4] == b"\x00\x00\x80\x28"]
print("existing 2^-46 (0x28800000) constants:", [hex(x) for x in new][:10])
# symbols
syms = []
for line in subprocess.run(["nm", "-C", "-S", "--defined-only", L], capture_output=True, text=True).stdout.splitlines():
    p = line.split(" ", 3)
    if len(p) == 4 and p[2] in "tTwW":
        try: syms.append((int(p[0], 16), int(p[1], 16), p[3]))
        except ValueError: pass
syms.sort(); starts = [s[0] for s in syms]
def sym(a):
    i = bisect.bisect_right(starts, a) - 1
    if i >= 0 and syms[i][0] <= a < syms[i][0] + max(syms[i][1], 1): return syms[i]
    return (0, 0, "?")
tva, toff, tsize = secs[".text"]
dis = subprocess.run(["objdump", "-d", "-M", "intel", "--no-show-raw-insn", "-j", ".text", L], capture_output=True, text=True).stdout.splitlines()
ins = []  # (addr, text)
for l in dis:
    m = re.match(r"\s*([0-9a-f]+):\s+(.*)", l)
    if m: ins.append((int(m.group(1), 16), m.group(2)))
print("instructions", len(ins))
hits = []
for k, (a, t) in enumerate(ins):
    m = re.search(r"\[rip\+0x([0-9a-f]+)\]\s+#\s+([0-9a-f]+)", t)
    if m and int(m.group(2), 16) in eps_addrs: hits.append(k)
print("instructions referencing an epsilon constant:", len(hits))
byfn = {}
for k in hits:
    a, t = ins[k]; s = sym(a); byfn.setdefault(s[2], []).append(k)
res = []
for fn, ks in sorted(byfn.items(), key=lambda x: x[0]):
    kinds = [ins[k][1].split()[0] for k in ks]
    print(f"\n{len(ks):3d} {fn[:150]}  {dict((x, kinds.count(x)) for x in set(kinds))}")
    for k in ks: res.append({"fn": fn, "addr": ins[k][0], "ins": ins[k][1], "k": k})
if out_json:
    json.dump({"hits": res, "eps_addrs": sorted(eps_addrs)}, open(out_json, "w"))
    # context dump for functions of interest
    with open(out_json.replace(".json", "_ctx.txt"), "w") as f:
        for fn, ks in sorted(byfn.items()):
            if not re.search(r"gjk|closestPt|EPA|Gjk|GJK|sweep|Sweep|raycast|Raycast|pcmContact|computeMTD", fn): continue
            for k in ks:
                f.write(f"\n=== {fn[:160]} @ {ins[k][0]:x}\n")
                for j in range(max(0, k - 14), min(len(ins), k + 10)): f.write(f"{'>>' if j == k else '  '} {ins[j][0]:x}: {ins[j][1]}\n")
