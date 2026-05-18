#!/bin/bash

MODE=${1:-train}
shift

if [[ "$MODE" == "train" ]]; then
  echo "Running training and testing..."
  python main.py "$@"

elif [[ "$MODE" == "test" ]]; then
  if [[ ! -f "best_model.pth" ]]; then
    echo "Error: best_model.pth not found. Please run training first or provide a pre-trained model."
    exit 1
  fi
  echo "Running evaluation with pre-trained model..."
  python main.py --skip-train "$@"

else
  echo "Invalid mode specified. Use 'train' or 'test'."
  exit 1
fi