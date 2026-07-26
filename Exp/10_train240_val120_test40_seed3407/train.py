from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
from common import SplitRunSpec, run_split_experiment

RUN_NAME = "10_train240_val120_test40_seed3407"
RATIO = "6:3:1"
SPLIT_SEED = 3407
TRAINING_SEED = 3407
TRAIN_COUNT = 240
VAL_COUNT = 120
TEST_COUNT = 40

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 10_train240_val120_test40_seed3407")
    parser.add_argument("--resume", action="store_true", help="Resume this exact frozen split after interruption.")
    args = parser.parse_args()
    spec = SplitRunSpec(
        RUN_NAME,
        RATIO,
        SPLIT_SEED,
        TRAINING_SEED,
        TRAIN_COUNT,
        VAL_COUNT,
        TEST_COUNT,
    )
    run_split_experiment(spec, Path(__file__).resolve().parent, resume=args.resume)
