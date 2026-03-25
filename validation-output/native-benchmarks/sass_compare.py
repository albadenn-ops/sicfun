"""Compare SASS instruction counts between old and optimized GPU DLLs."""
import re, os

TMPDIR = os.environ.get("TEMP", r"C:\Users\MK1\AppData\Local\Temp")

def count_per_kernel(filepath):
    kernels = {}
    current = None
    with open(filepath) as f:
        for line in f:
            m = re.match(r"\s+Function : (.+)", line)
            if m:
                current = m.group(1).strip()
                kernels[current] = {"total": 0, "bra": 0, "setp": 0, "shift": 0}
            elif current and re.match(r"\s+/\*[0-9a-f]+\*/", line):
                kernels[current]["total"] += 1
                if re.search(r"\b(BRA|BRX|SSY|SYNC)\b", line):
                    kernels[current]["bra"] += 1
                if re.search(r"\b(ISETP|FSETP|ISET|FSET|SEL)\b", line):
                    kernels[current]["setp"] += 1
                if re.search(r"\b(SHL|SHR|SHF|LOP|LOP3|BFE|BFI|POPC)\b", line):
                    kernels[current]["shift"] += 1
    return kernels

old = count_per_kernel(os.path.join(TMPDIR, "old_sass.txt"))
new = count_per_kernel(os.path.join(TMPDIR, "optimized_sass.txt"))

name_map = [
    ("exact_kernel_packed", "exact_packed"),
    ("exact_prepare_endpoint", "exact_prepare"),
    ("exact_accumulate", "exact_accum"),
    ("monte_carlo_kernel_packed_parallel", "mc_packed_par"),
    ("monte_carlo_kernel_packed", "mc_packed"),
    ("monte_carlo_kernel_parallel", "mc_par_trials"),
    ("range_monte_carlo_csr_by_hero_kernelILb0", "range_mc_nobias"),
    ("range_monte_carlo_csr_by_hero_kernelILb1", "range_mc_bias"),
    ("exact_kernel", "exact_dbl"),
    ("monte_carlo_kernel", "mc_dbl"),
]

def friendly(mangled):
    for pat, name in name_map:
        if pat in mangled:
            return name
    return mangled[:20]

def ds(o, n):
    d = n - o
    if o == 0:
        return f"{n:>5}   NEW"
    pct = d * 100.0 / o
    return f"{n:>5} {d:>+5} ({pct:+5.1f}%)"

all_k = sorted(
    set(list(old.keys()) + list(new.keys())),
    key=lambda k: old.get(k, {}).get("total", 0),
    reverse=True,
)

hdr = f"{'Kernel':<18} | {'Instructions':>28} | {'Branches':>28} | {'Cmp/Set':>28} | {'Bitwise':>28}"
sub = f"{'':18} | {'old':>5} {'new  delta (%)':>22} | {'old':>5} {'new  delta (%)':>22} | {'old':>5} {'new  delta (%)':>22} | {'old':>5} {'new  delta (%)':>22}"
print(hdr)
print(sub)
print("-" * len(hdr))

tot = dict(ot=0, nt=0, ob=0, nb=0, os=0, ns=0, osh=0, nsh=0)
for k in all_k:
    o = old.get(k, {"total": 0, "bra": 0, "setp": 0, "shift": 0})
    n = new.get(k, {"total": 0, "bra": 0, "setp": 0, "shift": 0})
    name = friendly(k)
    tot["ot"] += o["total"]; tot["nt"] += n["total"]
    tot["ob"] += o["bra"]; tot["nb"] += n["bra"]
    tot["os"] += o["setp"]; tot["ns"] += n["setp"]
    tot["osh"] += o["shift"]; tot["nsh"] += n["shift"]
    print(
        f"{name:<18} | {o['total']:>5} {ds(o['total'],n['total']):>22}"
        f" | {o['bra']:>5} {ds(o['bra'],n['bra']):>22}"
        f" | {o['setp']:>5} {ds(o['setp'],n['setp']):>22}"
        f" | {o['shift']:>5} {ds(o['shift'],n['shift']):>22}"
    )

print("-" * len(hdr))
print(
    f"{'TOTAL':<18} | {tot['ot']:>5} {ds(tot['ot'],tot['nt']):>22}"
    f" | {tot['ob']:>5} {ds(tot['ob'],tot['nb']):>22}"
    f" | {tot['os']:>5} {ds(tot['os'],tot['ns']):>22}"
    f" | {tot['osh']:>5} {ds(tot['osh'],tot['nsh']):>22}"
)
