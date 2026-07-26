from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
from common import SplitRunSpec, run_split_experiment

RUN_NAME = "03_train320_val40_test40_seed1234"
RATIO = "8:1:1"
SPLIT_SEED = 1234
TRAINING_SEED = 1234
TRAIN_COUNT = 320
VAL_COUNT = 40
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
