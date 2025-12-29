#!/bin/bash
# Master script to generate ALL project figures
# Usage: ./scripts/generate_all_figures.sh

set -e  # Exit on error

# Ensure we are in project root
if [ ! -d "scripts" ]; then
    echo "Error: Please run this script from the project root directory."
    exit 1
fi

export PYTHONPATH=.

echo "========================================================"
echo "Generating All Figures..."
echo "========================================================"

# 1. Class Balance
echo "[1/8] Plotting Class Balance..."
python scripts/plot_class_balance.py

# 2. Efficiency Curve
echo "[2/8] Plotting Efficiency Curve..."
python scripts/plot_efficiency.py

# 3. Phase A Heatmaps
echo "[3/8] Plotting Phase A Heatmaps..."
python scripts/plot_phase_a_heatmap.py

# 4. Phase B Stability
echo "[4/8] Plotting Phase B Stability..."
python scripts/plot_phase_b_stability.py

# 5. Phase C History
echo "[5/8] Plotting Phase C Search History..."
python scripts/plot_phase_c_history.py

# 6. Phase D Ablation & Table
echo "[6/8] Plotting Phase D Ablation & Generating Table..."
python scripts/plot_phase_d_ablation.py

# 7. Phase D Comprehensive (Robustness/Overfitting)
echo "[7/8] Plotting Phase D Comprehensive Analysis..."
python scripts/plot_phase_d_comprehensive.py

# 8. Augmentation Visualization
echo "[8/8] Visualizing Augmentation Examples..."
python scripts/visualize_augmentations.py

echo "========================================================"
echo "SUCCESS: All figures generated in outputs/figures/"
echo "========================================================"
