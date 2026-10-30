# What can Spark do for weather and climate?

Rough guide to workloads. "Expected" means *expected from specs and model sizes, not measured*.
Measured numbers for many of these are in the [one-pager](05-hardware-one-pager-spark-vs-cloud.md).

| Workload | Expected on Spark | Why |
|---|---|---|
| Run a single global AI forecast (e.g. SFNO, FourCastNet-class, Pangu-class), 0.25°, 10-day | Comfortable | Models are 0.1–2 B params; fits in memory many times over |
| Ensemble forecasts (tens of members) | Feasible, time scales with members | Memory is ample; compute is the limit, not capacity |
| Large foundation models (Aurora-class, AIFS) | Feasible | Memory headroom is the Spark advantage |
| Regional downscaling (diffusion, e.g. CorrDiff) | Feasible but slower | Iterative sampling is compute- and bandwidth-heavy |
| Nowcasting (StormScope/StormCast-class) | Feasible for small domains | Depends on domain size and GPU kernel availability |
| Data assimilation | Test per model | Depends on custom CUDA ops |
| Training / large fine-tuning | Not the target | Use cloud or DGX/HGX |
| Serving a forecast API to many users | Light load only | Single GPU, shared memory |

## What makes it useful for this training

- **Everyone gets a real environment.** SSH in, run Jupyter, pull the same containers used in the cloud.
- **Data stays local.** National met services can run models on their own data without uploading it.
- **Cost-predictable.** No per-hour GPU bill during a workshop.
- **Honest hardware lesson.** Participants see what a realistic "small institution" setup can and cannot do,
  which is exactly the decision they face back home.

Next: [best practices](03-best-practices.md).
