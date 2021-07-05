#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set -euo pipefail
conda activate gray
exec python /app/image_gray.py
