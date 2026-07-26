from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
from common import SplitRunSpec, run_split_experiment

RUN_NAME = "05_train280_val80_test40_seed42"
RATIO = "7:2:1"
SPLIT_SEED = 42
TRAINING_SEED = 42
TRAIN_COUNT = 280
VAL_COUNT = 80
TEST_COUNT = 40

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
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
