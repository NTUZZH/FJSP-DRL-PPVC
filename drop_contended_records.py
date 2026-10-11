"""Remove time-limited solver records that were measured under CPU contention.

A record whose cpu_share (CPU seconds / (wall seconds x workers)) is below the
threshold did not have its cores to itself, so its objective is not a valid
time-limited result. This script moves such records out of an anytime JSONL
file into a side file, so that the resumable runner re-solves those instances
on the next run. Never run it on a file that a solver process is appending to.

    python drop_contended_records.py or_solution/PPVC/anytime/<file>.jsonl [--threshold 0.9] [--min_solve_s 10]

A solve that ends in under --min_solve_s seconds (an instance proven optimal
almost immediately) is kept whatever its cpu_share: its share is dominated by
single-threaded start-up, not by contention, and its objective is a proven
optimum that no contention can change.
"""
import argparse
import json
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--threshold', type=float, default=0.9)
    ap.add_argument('--min_solve_s', type=float, default=10.0)
    args = ap.parse_args()
    keep, drop = [], []
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            short = r.get('solve_elapsed_s', float('inf')) < args.min_solve_s
            (keep if short or r.get('cpu_share', 0.0) >= args.threshold else drop).append(r)
    if not drop:
        print(f'{args.path}: {len(keep)} records, none below {args.threshold}')
        return 0
    backup = args.path + '.contended.jsonl'
    with open(backup, 'a') as f:
        for r in drop:
            f.write(json.dumps(r) + '\n')
    tmp = args.path + '.tmp'
    with open(tmp, 'w') as f:
        for r in keep:
            f.write(json.dumps(r) + '\n')
    shutil.move(tmp, args.path)
    ids = [r['instance_id'] for r in drop]
    print(f'{args.path}: kept {len(keep)}, moved {len(drop)} contended record(s) {ids} to {os.path.basename(backup)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
