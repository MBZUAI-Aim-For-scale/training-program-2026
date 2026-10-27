# %% [markdown]
# # Lab 02 — What is the GPU good at?
# **Goal:** measure matmul throughput by precision and memory bandwidth, and see why Spark is strong on
# capacity and compute but not on raw bandwidth.
#
# Expected time: 10 min. No downloads.

# %%
import time, torch
assert torch.cuda.is_available()
dev = "cuda"

def timeit(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# %% [markdown]
# ## Matmul throughput by precision
# TFLOPS = 2·N³ / time. TF32 needs the matmul precision flag; fp32 below is strict fp32.

# %%
N = 8192
results = {}
for name, dtype, tf32 in [("fp32", torch.float32, False), ("tf32", torch.float32, True),
                          ("bf16", torch.bfloat16, False), ("fp16", torch.float16, False)]:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(N, N, device=dev, dtype=dtype)
    b = torch.randn(N, N, device=dev, dtype=dtype)
    t = timeit(lambda: a @ b)
    results[name] = 2 * N**3 / t / 1e12
    print(f"{name}: {results[name]:7.1f} TFLOPS")
    del a, b
torch.backends.cuda.matmul.allow_tf32 = False

# %% [markdown]
# ## Memory bandwidth
# Copying a large tensor reads and writes each byte once. Compare with the published ~273 GB/s.

# %%
nbytes = 4 * 2**30
x = torch.empty(nbytes // 4, device=dev, dtype=torch.float32).normal_()
y = torch.empty_like(x)
t = timeit(lambda: y.copy_(x), iters=10)
bw = 2 * nbytes / t / 1e9
print(f"copy bandwidth: {bw:.0f} GB/s")
del x, y

# %% [markdown]
# ## Plot

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].bar(results.keys(), results.values(), color="#76B900")
ax[0].set_ylabel("TFLOPS"); ax[0].set_title(f"matmul {N}x{N}")
ax[1].bar(["copy"], [bw], color="#555"); ax[1].axhline(273, ls="--", c="r"); ax[1].set_ylabel("GB/s")
ax[1].set_title("memory bandwidth (red = spec)")
plt.tight_layout(); plt.savefig("outputs/lab02_benchmark.png", dpi=120); plt.show()

# %% [markdown]
# ## Discussion
# - Which precision is fastest? Which of your forecast models could run in bf16/fp16?
# - Is bandwidth near the spec? Weather models over 0.25° grids are largely bandwidth-bound:
#   what does that predict about runtime vs an H100?
# - Quantify: a 1B-parameter model in bf16 is 2 GB. How many fit in memory? (Lab 04 does this.)
