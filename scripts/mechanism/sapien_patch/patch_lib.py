"""Patch a COPY of SAPIEN's libsapien.so (PhysX 5.3.1): the triangle degeneracy test of the GJK simplex solver,
`if(FAllGrtrOrEq(eps, area))` with eps = FLT_EPSILON, is made to use PX_EPS_REAL^2 = 2^-46 as PhysX 5.4.0 does.
Each site is `movss xmmN, [rip+disp] ; comiss xmmN, xmm_area`; only the disp32 is changed so the load reads a new
constant placed in the padding after .fini (the shared FLT_EPSILON constant itself is untouched).
usage: patch_lib.py ORIG OUT LOG.json"""
import sys, struct, json, subprocess, re, os
orig, out, log = sys.argv[1:4]
b = bytearray(open(orig, "rb").read())
SITES = {0x3e0f07: "physx::Gu::GJKCPairDoSimplex(Vec3V*, Vec3V*, Vec3V*, const Vec3V&, PxU32&)",
         0x3eaec5: "physx::Gu::GJKCPairDoSimplex(Vec3V*, Vec3V*, Vec3V*, PxI32*, PxI32*, const Vec3V&, PxU32&)",
         0x43f8b0: "physx::Gu::closestPtPointTetrahedron(Vec3V*, Vec3V*, Vec3V*, PxU32&)",
         0x43ff79: "physx::Gu::closestPtPointTetrahedron(Vec3V*, Vec3V*, Vec3V*, PxI32*, PxI32*, PxU32&)"}
EPS_ADDR = 0x57d900; NEW_ADDR = 0x577720            # file offset == vaddr in this LOAD segment (0x89000 -> 0x89000)
assert struct.unpack_from("<I", b, EPS_ADDR)[0] == 0x34000000, "FLT_EPSILON not where expected (rodata offset == vaddr)"
assert bytes(b[0x577705:0x578000]) == bytes(0x578000 - 0x577705), "padding after .fini is not all zero"
# program header of the executable LOAD segment
e_phoff, = struct.unpack_from("<Q", b, 0x20); e_phentsize, e_phnum = struct.unpack_from("<HH", b, 0x36)
ph = None
for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack_from("<IIQQQQQQ", b, off)
    if p_type == 1 and p_offset == 0x89000 and (p_flags & 1): ph = (off, p_filesz, p_memsz, p_vaddr)
assert ph and ph[1] == 0x4ee705 and ph[2] == 0x4ee705 and ph[3] == 0x89000, ph
records = []
struct.pack_into("<4I", b, NEW_ADDR, *([0x28800000] * 4))        # 2^-46 = 1.4210855e-14, four times (16 bytes)
records.append({"what": "new constant 4 x float 2^-46", "file_offset": hex(NEW_ADDR), "old": "00" * 16, "new": bytes(b[NEW_ADDR:NEW_ADDR + 16]).hex()})
new_size = NEW_ADDR + 16 - 0x89000
old_ph = bytes(b[ph[0] + 0x20: ph[0] + 0x30]).hex()
struct.pack_into("<QQ", b, ph[0] + 0x20, new_size, new_size)
records.append({"what": "executable LOAD segment p_filesz/p_memsz 0x4ee705 -> " + hex(new_size), "file_offset": hex(ph[0] + 0x20), "old": old_ph, "new": bytes(b[ph[0] + 0x20: ph[0] + 0x30]).hex()})
for addr, fn in sorted(SITES.items()):
    ins = bytes(b[addr: addr + 11])
    assert ins[0:3] == b"\xf3\x0f\x10" and (ins[3] & 0xc7) == 0x05, (hex(addr), ins.hex())          # movss xmmN, [rip+disp32]
    reg = (ins[3] >> 3) & 7
    disp, = struct.unpack_from("<i", ins, 4)
    assert addr + 8 + disp == EPS_ADDR, (hex(addr), hex(addr + 8 + disp))
    assert ins[8:10] == b"\x0f\x2f" and ((ins[10] >> 3) & 7) == reg and (ins[10] & 0xc0) == 0xc0, (hex(addr), ins.hex())   # comiss xmmN, xmmM
    nd = NEW_ADDR - (addr + 8)
    struct.pack_into("<i", b, addr + 4, nd)
    records.append({"what": "movss xmm%d,[rip+disp]: 0x57d900 (FLT_EPSILON) -> 0x577720 (2^-46); next: comiss xmm%d,xmm%d" % (reg, reg, ins[10] & 7),
                    "function": fn, "vaddr": hex(addr), "file_offset": hex(addr + 4), "old": ins[4:8].hex(), "new": bytes(b[addr + 4: addr + 8]).hex()})
tmp = out + ".tmp"; open(tmp, "wb").write(b); os.chmod(tmp, os.stat(orig).st_mode); os.replace(tmp, out)
json.dump(records, open(log, "w"), indent=1)
diff = sum(1 for x, y in zip(open(orig, "rb").read(), b) if x != y)
print("patched", len(SITES), "sites; bytes differing from the original:", diff)
for r in records: print(r)
