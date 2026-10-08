"""Build data_npz/ (compact integer mirror) from the TSV release in data/."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, DST = os.path.join(ROOT, "data"), os.path.join(ROOT, "data_npz")
COLS = ["user_id", "item_id", "timestamp", "correct", "skill_id"]
DTYPE = dict(user_id=np.int32, item_id=np.int32, timestamp=np.int64,
             correct=np.int8, skill_id=np.int32)

if __name__ == "__main__":
    total = 0
    for ds in sorted(os.listdir(SRC)):
        os.makedirs(os.path.join(DST, ds), exist_ok=True)
        for split in ("train", "test"):
            src = os.path.join(SRC, ds, f"preprocessed_data_{split}.csv")
            if not os.path.exists(src):
                continue
            df = pd.read_csv(src, sep="\t")
            out = os.path.join(DST, ds, f"{split}.npz")
            np.savez_compressed(out, **{c: df[c].to_numpy(DTYPE[c]) for c in COLS})
            total += os.path.getsize(out)
        print("  ok ", ds)
    print(f"data_npz total: {total / 1048576:.1f} MB")