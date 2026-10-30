# %% [markdown]
# # Lab 01 — Know your Spark
# **Goal:** in 15 minutes, learn the three numbers that decide what you can run on a DGX Spark for weather work:
# how much memory you have (128 GB, shared by CPU and GPU), how fast the GPU computes, and how fast it reads
# memory (the real limit for AI weather models).
#
# Expected time: 15 min. No downloads.

# %%
import os, platform, re, subprocess, time
import torch

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()

assert torch.cuda.is_available(), "No CUDA GPU visible. Are you inside a GPU container (--device nvidia.com/gpu=all)?"
major, minor = torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info()
print("Arch       :", platform.machine(), "(Arm, not x86: some Python wheels don't exist for it)")
print("CPU cores  :", os.cpu_count())
print("GPU        :", torch.cuda.get_device_name(0), f"sm_{major}{minor}")
print(f"Memory     : CUDA sees {total/2**30:.0f} GiB total, {free/2**30:.0f} GiB free")
print(sh("free -g | head -2"))
print("torch      :", torch.__version__, "| CUDA", torch.version.cuda)

# %% [markdown]
# ## 1. One memory pool
# `free` (what Linux sees) and CUDA's total are nearly the same number: CPU and GPU share it. That is why a
# 50 GB model fits here when it wouldn't on many data-centre GPUs, and also why a big CPU job (or the file
# cache) leaves less room for the GPU.
#
# ## 2. Does this PyTorch have kernels for this GPU?
# The most common Spark pitfall is a PyTorch build with no code for this GPU ("no kernel image is available").
# A kernel built for `sm_120` runs on this `sm_121`; one built only for `sm_90` (Hopper) does not.

# %%
def runs_here(arch):
    m = re.fullmatch(r"(sm|compute)_(\d+)(\d)([af]?)", arch)
    if not m:
        return False
    kind, x, y, suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    if suffix == "a":
        return (x, y) == (major, minor)
    if kind == "sm" or suffix == "f":
        return x == major and y <= minor
    return (x, y) <= (major, minor)

usable = [a for a in torch.cuda.get_arch_list() if runs_here(a)]
print("usable kernels in this torch build:", usable or "NONE: use an NGC PyTorch container")

# %% [markdown]
# ## 3. How fast is it? Compute vs memory bandwidth

# %%
def timeit(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

N, tflops = 8192, {}
for name, dtype, tf32 in [("fp32", torch.float32, False), ("tf32", torch.float32, True),
                          ("bf16", torch.bfloat16, False)]:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(N, N, device="cuda", dtype=dtype); b = torch.randn_like(a)
    tflops[name] = 2 * N**3 / timeit(lambda: a @ b) / 1e12
    del a, b
torch.backends.cuda.matmul.allow_tf32 = False

x = torch.empty(2**30, device="cuda").normal_(); y = torch.empty_like(x)   # 4 GiB each
bw = 2 * x.nbytes / timeit(lambda: y.copy_(x), iters=10) / 1e9
del x, y
for k, v in tflops.items():
    print(f"{k}: {v:5.1f} TFLOPS")
print(f"memory bandwidth: {bw:.0f} GB/s (spec ~273)")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].bar(tflops.keys(), tflops.values(), color="#76B900"); ax[0].set_ylabel("TFLOPS"); ax[0].set_title("compute")
ax[1].bar(["copy"], [bw], color="#555"); ax[1].axhline(273, ls="--", c="r"); ax[1].set_ylabel("GB/s")
ax[1].set_title("memory bandwidth (red = spec)")
plt.tight_layout(); plt.savefig("outputs/lab01_benchmark.png", dpi=120); plt.show()

# %% [markdown]
# **What this means for weather models:** each forecast step reads all the model's weights and the whole
# atmosphere state from memory. With ~230 GB/s that, not compute, sets the pace: a data-centre GPU (3–8 TB/s)
# is 10–30x faster per step. The Spark wins on *room* (big models, ensembles) and on being yours, not on speed.
#
# ## 4. Will it fit? Size a forecast before you run it
# A 0.25° global grid is 721 x 1440 points. One variable at one time in fp32 is ~4 MB.

# %%
def state_gib(channels, members=1):
    return channels * 721 * 1440 * 4 * members / 2**30

BUDGET_GIB = 110   # leave headroom for the OS, Jupyter and other users
print(f"{'model':<28} {'weights':>8} {'state':>8}")
for name, params_b, channels in [("SFNO (Lab 02)", 0.4, 73), ("AIFS-ENS (Lab 04)", 0.25, 100),
                                 ("Aurora (large)", 1.3, 70)]:
    print(f"{name:<28} {params_b*2:>6.1f}GB {state_gib(channels):>6.2f}GB")
print("\nMeasured on this Spark: AIFS-ENS, 8 members, 5 days (Lab 04) peaked at 36 GiB of GPU memory:")
print("members run one after another, so memory does not grow with ensemble size, only time does.")
print("Activations inside a model are many times the state: always measure (torch.cuda.max_memory_allocated).")

# %% [markdown]
# ## 5. Sharing the box
# Teams usually get one container each with `docker run --memory 90g ...`. On the Spark **that limit does not
# cap GPU memory**: we verified a container limited to 8 GiB could fill 16 GiB on the GPU (the same amount on
# the CPU was killed). The heaviest notebooks peak at 70–77 GiB, so **schedule heavy jobs: one at a time**.
# Check before you start:

# %%
print(sh("free -g | head -2"))
print(sh("docker ps --format '{{.Names}}  {{.Status}}' 2>/dev/null") or "(docker not visible inside the container; run `docker ps` on the host)")

# %% [markdown]
# ## Exercise
# 1. Lab 04 runs members one at a time (36 GiB). If you batched 4 members at once, roughly what would it need,
#    and would it fit in 110 GiB with an LLM (16 GiB, Lab 05) loaded too?
# 2. bf16 is ~5x faster than fp32 here. Why might a forecast model still run in fp32?
# 3. If each forecast step must read ~2 GB of weights + state, what is the fastest a step can be at the
#    bandwidth you measured?
