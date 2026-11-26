"""ON vs OFF balance of the raw events and of the EvDownsampling input (from the cache).

    python polarity_balance.py DATA_ROOT --cache-dir CACHE_DIR
"""
import argparse
from pathlib import Path

import numpy as np
from tonic import transforms
from tqdm import tqdm

from data import (DENOISE_US, EVDOWNSAMPLING_PARAMS, NUM_CLASSES, crop_events, encode_recordings,
                  load_dataset, split_train_validation)


def on_share(on, off):
    return on / np.maximum(on + off, 1)


def report(name, on, off, labels):
    share = on_share(on, off)
    print(f"  {name:<15} ON {on.sum():>11,}  OFF {off.sum():>11,}  "
          f"ON share {on.sum() / (on.sum() + off.sum()):.3f}  "
          f"(per recording: median {np.median(share):.3f}, "
          f"5-95% {np.percentile(share, 5):.3f}-{np.percentile(share, 95):.3f})")
    per_class = [on[labels == c].sum() / max(on[labels == c].sum() + off[labels == c].sum(), 1)
                 for c in range(NUM_CLASSES)]
    print(f"  {'':<15} ON share per class: " + " ".join(f"{c}:{s:.2f}"
                                                         for c, s in enumerate(per_class)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--crop-ms", type=int, default=4000)
    parser.add_argument("--evd-params", type=float, nargs=3, default=EVDOWNSAMPLING_PARAMS,
                        metavar=("TAU_THETA", "TAU_ACC", "BETA"))
    args = parser.parse_args()

    dataset = load_dataset(args.data_root.resolve(), train=True)
    train_idx, val_idx = split_train_validation(dataset)
    for split, indices in (("train", train_idx), ("validation", val_idx)):
        spikes, labels, _ = encode_recordings(dataset, indices, "evdownsampling",
                                              f"Loading {split}", None,
                                              args.cache_dir.resolve(), args.crop_ms,
                                              args.evd_params)
        # even ids are OFF, odd are ON
        counts = [np.diff(s.end_spikes, prepend=0) for s in spikes]
        evd_on = np.array([c[1::2].sum() for c in counts])
        evd_off = np.array([c[0::2].sum() for c in counts])

        raw_on, raw_off = np.zeros(len(indices), int), np.zeros(len(indices), int)
        for j, i in enumerate(tqdm(indices, desc=f"Raw events {split}", unit="recording")):
            events = crop_events(dataset[i][0], args.crop_ms)
            events = transforms.Denoise(filter_time=DENOISE_US)(events)
            raw_on[j] = np.count_nonzero(events["p"] == 1)
            raw_off[j] = np.count_nonzero(events["p"] == 0)

        print(f"\n{split}: {len(indices)} recordings, first {args.crop_ms} ms")
        report("raw events", raw_on, raw_off, labels)
        report("EvDownsampling", evd_on, evd_off, labels)
        kept = (evd_on.sum() + evd_off.sum()) / (raw_on.sum() + raw_off.sum())
        print(f"  EvDownsampling keeps {kept:.1%} of the raw events "
              f"(ON {evd_on.sum() / raw_on.sum():.1%}, OFF {evd_off.sum() / raw_off.sum():.1%})")


if __name__ == "__main__":
    main()
