# %% [markdown]
# # Lab 04 — Memory budgeting and sizing (best practices)
# **Goal:** learn to estimate whether a model and ensemble fit before you run, and see how unified memory behaves.
#
# Expected time: 10 min. No downloads.

# %% [markdown]
# ## 1. Model size calculator
# Weights only. Activations and ensemble members add more; see section 2.

# %%
BYTES = {"fp32": 4, "bf16/fp16": 2, "fp8": 1, "fp4": 0.5}

def weights_gib(params_billion):
    return {k: params_billion * 1e9 * b / 2**30 for k, b in BYTES.items()}

print(f"{'params':>8} " + " ".join(f"{k:>10}" for k in BYTES))
for p in [0.1, 0.5, 1.3, 7, 70, 200]:
    w = weights_gib(p)
    print(f"{p:>6}B  " + " ".join(f"{w[k]:>8.1f}Gi" for k in BYTES))

BUDGET_GIB = 110  # 128 GB minus OS/Jupyter headroom (see docs/03-best-practices.md)
print(f"\nBudget used here: {BUDGET_GIB} GiB")

# %% [markdown]
# ## 2. Grid-state size: what one forecast state costs
# A 0.25° global grid is 721 x 1440. With C channels in fp32, one state is `C * 721 * 1440 * 4` bytes.

# %%
H, W = 721, 1440
def state_gib(channels, members=1, dtype_bytes=4):
    return channels * H * W * dtype_bytes * members / 2**30

for C in [73, 100, 200]:
    print(f"{C:>3} channels: {state_gib(C):.2f} GiB per member | 50 members: {state_gib(C, 50):.1f} GiB")
print("Activations inside the model are typically many times the state size. Always measure.")

# %% [markdown]
# ## 3. Measure: allocate until you hit the budget
# On unified memory the CUDA "free" figure shrinks as the CPU uses RAM and page cache too.

# %%
import torch
assert torch.cuda.is_available()

def free_gib():
    return torch.cuda.mem_get_info()[0] / 2**30

start_free = free_gib()
print(f"CUDA free at start: {start_free:.1f} GiB")
# Stay well clear of the limit: the box is shared and running out of unified memory can stall the whole system.
target = int(min(32, start_free - 20) // 4)
print(f"allocating {target} x 4 GiB")
held = []
try:
    for i in range(1, target + 1):
        held.append(torch.empty(4 * 2**30 // 4, device="cuda"))  # 4 GiB each
        print(f"allocated {4*i:>3} GiB -> CUDA free {free_gib():6.1f} GiB")
finally:
    del held
    torch.cuda.empty_cache()
print(f"after release: {free_gib():.1f} GiB")

# %% [markdown]
# ## 4. Page cache experiment
# Read a large file, then re-check CUDA free memory. Linux file cache counts against it until reclaimed.
# Run in a terminal (needs a big file, e.g. downloaded weights):
# ```bash
# free -h && cat ~/.cache/earth2studio/*/* > /dev/null 2>&1; free -h
# sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'   # reclaims it
# ```

# %% [markdown]
# ## 5. Does a container memory limit cap GPU allocations? (instructor demo)
# The plan for sharing one Spark gives each team a container with `--memory`. On unified memory you might
# expect GPU allocations to count against that limit. They don't. Try it from the host:
# ```bash
# docker run --rm --device nvidia.com/gpu=all --memory 8g spark-intro \
#   python -c "import torch; x=[torch.ones(2**30, device='cuda') for _ in range(4)]; print('filled 16 GiB')"
# ```
# It prints: 16 GiB on the GPU under an 8 GiB limit. Swap `'cuda'` for `'cpu'` and the container is
# OOM-killed instead. So `--memory` only caps CPU-side RAM; teams must coordinate heavy GPU jobs by schedule.
# (Verified on GB10, driver 580.95, NGC PyTorch 26.06; see `docs/03-best-practices.md`.)

# %% [markdown]
# ## Exercise
# 1. A 1.3 B-parameter model, 50-member ensemble, 5 GiB activations per member: fits in 110 GiB? Show the arithmetic.
# 2. Which ensemble size is the max if each member needs 2 GiB of state + 4 GiB of activations?
# 3. When is batching members on one GPU *faster* than running them sequentially? (hint: Lab 02 bandwidth)
