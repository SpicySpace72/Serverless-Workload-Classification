"""
Phase 1: Data Acquisition & Preprocessing
Semantic Workload Awareness in Serverless Computing

Run this file directly (VS Code Run button, or `python phase1_preprocess.py`
in the integrated terminal) from an empty folder — it downloads what it
needs and writes results into that same folder.

TEST_MODE runs a small, fast version first (2 days, 200 functions) so you
can confirm everything works before committing to the full 14-day,
10,000-function run, which takes longer and uses more memory.
"""

import os
import gc
import random
import urllib.request
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
TEST_MODE = False          # set False once a TEST_MODE run finishes cleanly
TEST_NUM_DAYS = 2
TEST_NUM_FUNCTIONS = 200

FULL_NUM_DAYS = 14
FULL_NUM_FUNCTIONS = 10_000

NUM_DAYS = TEST_NUM_DAYS if TEST_MODE else FULL_NUM_DAYS
NUM_FUNCTIONS_TO_SAMPLE = TEST_NUM_FUNCTIONS if TEST_MODE else FULL_NUM_FUNCTIONS

RANDOM_SEED = 42
MINUTES_PER_DAY = 1440
WINDOW_MINUTES = 15
CHUNK_SIZE = 50_000  # rows per chunk while reading each day's CSV

BASE_URL = "https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/invocations_per_function_md.anon.d{day:02d}.csv"
LOCAL_DATA_DIR = "."  # CSVs are downloaded into the same folder as this script

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def local_path(day):
    return os.path.join(LOCAL_DATA_DIR, f"invocations_per_function_md.anon.d{day:02d}.csv")


def ensure_downloaded(day):
    path = local_path(day)
    if os.path.exists(path):
        print(f"  Day {day}: already have it locally, skipping download.")
        return path
    url = BASE_URL.format(day=day)
    print(f"  Day {day}: downloading...")
    urllib.request.urlretrieve(url, path)
    return path


def main():
    print(f"TEST_MODE={TEST_MODE} | days={NUM_DAYS} | functions to sample={NUM_FUNCTIONS_TO_SAMPLE}\n")

    # --------------------------------------------------------
    # Step 1: Download the raw daily files (skips ones you already have)
    # --------------------------------------------------------
    print("Step 1: Downloading raw daily files...")
    for d in range(1, NUM_DAYS + 1):
        ensure_downloaded(d)

    # --------------------------------------------------------
    # PASS 1: Global 14-day invocation sum per function (chunked, low memory)
    # --------------------------------------------------------
    print("\nPass 1: Computing global invocation sums per function...")
    minute_cols = [str(i) for i in range(1, MINUTES_PER_DAY + 1)]
    global_sums = {}  # UID -> running total

    for d in range(1, NUM_DAYS + 1):
        path = local_path(d)
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            uid = (
                chunk["HashOwner"].astype(str) + "_" +
                chunk["HashApp"].astype(str) + "_" +
                chunk["HashFunction"].astype(str)
            )
            row_sums = chunk[minute_cols].sum(axis=1)
            for u, s in zip(uid, row_sums):
                global_sums[u] = global_sums.get(u, 0) + s
        print(f"  Day {d}: done. Unique functions seen so far: {len(global_sums):,}")
        gc.collect()

    active_uids = [u for u, total in global_sums.items() if total > 0]
    dead_count = len(global_sums) - len(active_uids)
    print(f"\nActive functions: {len(active_uids):,}  |  Dead functions filtered out: {dead_count:,}")

    if len(active_uids) < NUM_FUNCTIONS_TO_SAMPLE:
        raise ValueError(
            f"Only {len(active_uids)} active functions available, but "
            f"{NUM_FUNCTIONS_TO_SAMPLE} were requested. Lower NUM_FUNCTIONS_TO_SAMPLE "
            f"or raise NUM_DAYS."
        )

    sampled_uids = sorted(random.sample(active_uids, NUM_FUNCTIONS_TO_SAMPLE))
    uid_to_idx = {u: i for i, u in enumerate(sampled_uids)}
    n_sampled = len(sampled_uids)
    print(f"Sampled {n_sampled:,} functions (seed={RANDOM_SEED}, reproducible).")

    # --------------------------------------------------------
    # PASS 2: Extract full timelines for sampled functions only
    # Pre-allocated numpy array, NOT growing Python lists — this is the
    # part that previously risked ~8GB of overhead from list-of-floats.
    # --------------------------------------------------------
    print("\nPass 2: Building continuous timelines for sampled functions...")
    full_timeline = np.zeros((n_sampled, NUM_DAYS * MINUTES_PER_DAY), dtype=np.int32)
    # Missing days are already zero by construction (np.zeros), satisfying
    # the "impute missing minutes with 0" requirement automatically.

    for d in range(1, NUM_DAYS + 1):
        path = local_path(d)
        found_today = 0
        col_start = (d - 1) * MINUTES_PER_DAY
        col_end = col_start + MINUTES_PER_DAY

        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            uid = (
                chunk["HashOwner"].astype(str) + "_" +
                chunk["HashApp"].astype(str) + "_" +
                chunk["HashFunction"].astype(str)
            )
            mask = uid.isin(uid_to_idx.keys())
            if mask.any():
                matched_uid = uid[mask]
                matched_vals = chunk.loc[mask, minute_cols].values.astype(np.int32)
                for u, row in zip(matched_uid, matched_vals):
                    full_timeline[uid_to_idx[u], col_start:col_end] = row
                found_today += mask.sum()
        print(f"  Day {d}: matched {found_today:,} of {n_sampled:,} sampled functions")
        gc.collect()

    print(f"\nFull timeline matrix shape: {full_timeline.shape}  "
          f"({full_timeline.nbytes / 1024**2:.1f} MB)")

    # --------------------------------------------------------
    # Preprocessing: 15-minute tumbling windows (sum), then Min-Max normalize
    # --------------------------------------------------------
    print("\nDown-sampling into 15-minute tumbling windows...")
    n_windows = (NUM_DAYS * MINUTES_PER_DAY) // WINDOW_MINUTES
    windowed = full_timeline.reshape(n_sampled, n_windows, WINDOW_MINUTES).sum(axis=2)
    print(f"Windowed shape: {windowed.shape}")

    print("Applying Min-Max normalization per function...")
    mins = windowed.min(axis=1, keepdims=True)
    maxs = windowed.max(axis=1, keepdims=True)
    ranges = maxs - mins
    safe_ranges = np.where(ranges == 0, 1, ranges)
    normalized = (windowed - mins) / safe_ranges
    normalized = np.where(ranges == 0, 0, normalized)  # flat series -> all zeros, no div-by-zero

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------
    out_df = pd.DataFrame(
        normalized,
        index=sampled_uids,
        columns=[f"window_{i+1}" for i in range(n_windows)],
    )
    out_df.index.name = "UID"

    suffix = "TEST" if TEST_MODE else "full"
    save_path = os.path.join(LOCAL_DATA_DIR, f"preprocessed_{suffix}.csv")
    out_df.to_csv(save_path)

    print(f"\n{'TEST run' if TEST_MODE else 'FULL run'} complete!")
    print(f"Saved: {save_path}")
    print(f"Final matrix shape: {out_df.shape}")
    if TEST_MODE:
        print("\nIf this ran without errors, set TEST_MODE = False at the top "
              "of this file and run it again for the real 10,000-function dataset.")


if __name__ == "__main__":
    main()
