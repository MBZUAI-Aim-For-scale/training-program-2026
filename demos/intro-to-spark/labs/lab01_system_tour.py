# %% [markdown]
# # Lab 01 — Tour of your DGX Spark
# **Goal:** confirm what hardware and software you are on, and see the three Spark specifics:
# aarch64 CPU, unified memory, Blackwell (`sm_121`) GPU.
#
# Expected time: 10 min. No downloads.

# %%
import platform, os, shutil, subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()

print("Host      :", platform.node())
print("Arch      :", platform.machine(), "(expect aarch64 on Spark)")
print("OS        :", sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2"))
print("CPU cores :", os.cpu_count(), "(expect 20)")
print("CPU model :", sh("lscpu | grep -i 'model name' | head -2"))

# %% [markdown]
# ## Memory is unified
# CPU and GPU share one pool. Compare what Linux sees with what CUDA sees: they should be close.

# %%
print(sh("free -h"))
print()
print(sh("nvidia-smi --query-gpu=name,driver_version --format=csv"))
print("(nvidia-smi may show memory as N/A on unified-memory systems; that is expected)")

# %%
import torch
print("torch        :", torch.__version__)
print("CUDA runtime :", torch.version.cuda)
print("cuda avail   :", torch.cuda.is_available())
assert torch.cuda.is_available(), "No CUDA GPU visible. Are you inside a GPU container (--device nvidia.com/gpu=all)?"

free, total = torch.cuda.mem_get_info()
print(f"CUDA free/total: {free/2**30:.1f} / {total/2**30:.1f} GiB")
print("device       :", torch.cuda.get_device_name(0))
print("capability   :", torch.cuda.get_device_capability(0), "(Blackwell GB10 reports (12, 1))")
print("arch list    :", torch.cuda.get_arch_list())

# %% [markdown]
# ## Check: does this PyTorch have kernels for this GPU?
# PyTorch must ship code this GPU can run, otherwise CUDA ops fail with
# *"no kernel image is available for execution on the device"*. This is the most common Spark pitfall.
# Rules: a compiled `sm_XY` kernel runs on any GPU with the same major version X and minor >= Y
# (so `sm_120` runs on 12.1); `compute_XY` PTX can be JIT-compiled for any newer GPU;
# `sm_XYa` (arch-specific) kernels run only on exactly that GPU.

# %%
import re

def runs_here(arch, major, minor):
    m = re.fullmatch(r"(sm|compute)_(\d+)(\d)([af]?)", arch)
    if not m:
        return False
    kind, x, y, suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    if suffix == "a":
        return (x, y) == (major, minor)
    if kind == "sm" or suffix == "f":
        return x == major and y <= minor
    return (x, y) <= (major, minor)

major, minor = torch.cuda.get_device_capability(0)
usable = [a for a in torch.cuda.get_arch_list() if runs_here(a, major, minor)]
print(f"GPU sm_{major}{minor}; usable entries in this torch build: {usable or 'NONE'}")
print("->", "OK" if usable else "MISSING: use an NGC PyTorch container")

x = torch.randn(4096, 4096, device="cuda")
print("matmul smoke test:", (x @ x).sum().item() != 0)

# %% [markdown]
# ## Disk
# Model weights and ERA5 chunks are large. Know where your cache lives and how much room you have.

# %%
for p in ["/", os.path.expanduser("~")]:
    t, u, f = shutil.disk_usage(p)
    print(f"{p:<20} free {f/2**30:7.0f} GiB of {t/2**30:.0f} GiB")

# %% [markdown]
# ## Exercise
# 1. Record your results in a table: cores, RAM, CUDA free, capability, torch version.
# 2. Does `free -h` "total" match CUDA "total"? Why are they close but not equal?
# 3. In a terminal **on the Spark host** (not inside the container), run `docker ps` to see who else is
#    using the box, and `nvidia-smi` to see GPU processes.
