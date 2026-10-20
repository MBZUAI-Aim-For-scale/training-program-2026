---
title: Demo 5
teaching: 30
exercises: 15
questions:
- "How do we evaluate the AI weather forecasts we generated against different metrics?"
- "How can local analysis and publicly available datasets provide deeper insights into model performance?"
- "How do we prepare our model outputs for the upcoming ground truth validation?"
objectives:
- "Perform a customized evaluation of the forecast generated in the previous demo."
- "Compare models across different metrics using publicly available datasets accessed via Google Cloud Bucket."
- "Identify next steps and set up the discussion around ground truth validation based on use-case performance."
keypoints:
- "Customized evaluation against specific use-case metrics reveals the practical strengths and weaknesses of AI weather models."
- "Publicly available datasets accessed via cloud storage enable robust, reproducible model comparison."
- "Analyzing how models stand up to specific use cases is a critical prerequisite for meaningful ground truth validation."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Putting Your Forecast to the Test

## Customized Evaluation and Local Analysis Insights

**Audience:** All Participants (Use-Case Groups)  

### 1. Data Retrieval and Setup
- Open the provided Jupyter notebook environment.
- Connect to the Google Cloud Bucket to pull the necessary data.
- Retrieve the specific model outputs that your group generated and ran the day before.

### 2. Metric Evaluation
- Use-case groups will compare their models across different metrics using publicly available datasets.
- Distribute and review the **Discussion Sheet on Metrics** to guide the evaluation process.
- Examine how the models perform against these predefined metrics.

### 3. Deep Dive and Ground Truth Setup
- Conduct a deeper dive to present how the models stand up to your specific use case as inputs.
- Discuss what you want to do next based on these evaluation results.
- Use these insights to set up the foundation for the upcoming discussion around ground truth validation.