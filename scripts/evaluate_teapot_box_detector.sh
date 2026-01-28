#!/bin/bash
# Evaluate Teapot box detectors

# Activate virtual environment
source .venv/bin/activate

# Set default values
START_DATE=${1:-"2023-01-01"}
END_DATE=${2:-"2024-01-01"}
OUTPUT_DIR=${3:-"outputs/teapot/evaluation"}

echo "Evaluating Teapot box detectors..."
echo "Date range: $START_DATE to $END_DATE"
echo "Output directory: $OUTPUT_DIR"

# Run evaluation
python python/examples/teapot_box_detector_evaluation.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --output "$OUTPUT_DIR" \
    --per-stock \
    --use-cache

echo "Evaluation complete! Results saved to $OUTPUT_DIR"
