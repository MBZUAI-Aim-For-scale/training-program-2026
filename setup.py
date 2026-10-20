import os
from setuptools import setup, find_packages

# 1. Detect the nested 'momp' package
if os.path.isdir("apps/benchmarking/ROMP/momp"):
    momp_path = "apps/benchmarking/ROMP/momp"
elif os.path.isdir("ai_weather/apps/benchmarking/ROMP/momp"):
    momp_path = "ai_weather/apps/benchmarking/ROMP/momp"
else:
    raise FileNotFoundError("Could not find the 'momp' directory.")

momp_subpackages = find_packages(where=momp_path)

# 2. Find all other packages (apps) inside the 'ai_weather' folder
# This ensures `import apps` and `import ui` work globally
top_level_packages = find_packages(where="ai_weather")

setup(
    # Map the top-level 'momp' to its nested path
    # Map the "root" ("") to the 'ai_weather' folder.
    # This tells setuptools that top-level files like 'config.py' live inside 'ai_weather/'
    package_dir={
        "momp": momp_path,
        "": "ai_weather",
    },
    # Register all discovered packages
    packages=top_level_packages + ["momp"] + [f"momp.{p}" for p in momp_subpackages],
    # Register top-level .py files found inside the 'ai_weather' folder
    py_modules=["config", "dataset", "features", "plots", "utils"],
)