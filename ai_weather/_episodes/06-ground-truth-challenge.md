---
title: Demo 6
teaching: 30
exercises: 15
questions:
- "How does model performance change when validated against different observational datasets?"
- "How do AI models perform across key trigger thresholds like high and low rainfall events?"
- "How can we replicate this ground truth validation using our own national data?"
objectives:
- "Frame a commentary around ground truth and compare model performance across different observational datasets."
- "Benchmark models using preloaded data to explore performance differences for key trigger thresholds."
- "Discuss strategies for replicating this validation process with national data and ENACTS."
keypoints:
- "Model performance can vary significantly depending on the chosen observational ground truth dataset (e.g., ERA5 vs. CHIRPS vs. IMERG)."
- "Evaluating models against key trigger thresholds (high/low rainfall) is critical for practical service implementation."
- "Incorporating gauge-corrected data (like RFE corrected to rainfall gauges) improves the reliability of short-run rainfall evaluations."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Ground Truth Challenge 

## Validating Models with Different Obs Data

**Audience:** All Participants (Use-Case Groups)

### 1. Ground Truth Framing and Dataset Selection
- Introduce the core concepts of ground truth validation and distribute the **Discussion Sheet on Ground Truth Comparisons**.
- Instruct use-case groups to pull 2 distinct observational datasets specific to their use cases to compare how model performance changes.

### 2. Benchmarking and Threshold Analysis
- Use preloaded data to benchmark models and compare outputs across different observational datasets: ERA5, CHIRPS, IMERG, RFE, and MSWEP.
- **Special Addition:** Ensure the inclusion of RFE corrected to rainfall gauges specifically for the short-run rainfall use case.
- Explore and discuss differences in how the models perform for key trigger thresholds, with a specific focus on high and low rainfall events.

### 3. National Data Replication and ENACTS Readiness
- Discuss methodologies for how participants can later replicate this benchmarking process using their own national data.
- Tie the discussion back to the ENACTS readiness assessment and data construction covered in Demo 2 to ensure long-term sustainability.