#!/usr/bin/env python3
"""Plot per-task train/validation losses from a YOLO MTL tqdm log."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HEADER = re.compile(r"\b(train|val) epoch (\d+):.*?\|\s*(\d+)/(\d+)")
VALUE = re.compile(r"([A-Za-z][\w-]*)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def parse_log(path: Path):
    # tqdm rewrites a line with carriage returns. Keeping the final occurrence
    # for each batch removes those display duplicates.
    batches = {}
    totals = {}
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        header = HEADER.search(line)
        if not header:
            continue
        phase, epoch, step, total = header.groups()
        postfix_start = line.find("loss=", header.end())
        if postfix_start < 0:
            continue
        # Stop at tqdm's postfix closing bracket so distributed warnings such
        # as ``SocketImpl(fd=49)`` cannot be mistaken for loss metrics.
        postfix = line[postfix_start:].split("]", 1)[0]
        values = {name: float(value) for name, value in VALUE.findall(postfix)}
        if "loss" not in values:
            continue
        key = (phase, int(epoch), int(step))
        batches[key] = values
        totals[(phase, int(epoch))] = int(total)

    completed = {}
    for phase_epoch, total in totals.items():
        steps = [step for phase, epoch, step in batches if (phase, epoch) == phase_epoch]
        if steps and max(steps) >= total:
            completed[phase_epoch] = total

    histories = defaultdict(dict)
    metrics = set()
    for phase_epoch in sorted(completed):
        phase, epoch = phase_epoch
        rows = [values for (p, e, _), values in batches.items() if (p, e) == phase_epoch]
        metrics.update(name for row in rows for name in row)
        for metric in metrics:
            values = [row[metric] for row in rows if metric in row]
            if values:
                histories[(phase, metric)][epoch] = float(np.mean(values))
    return histories, ["loss", *sorted(metrics - {"loss"})]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    histories, metrics = parse_log(args.log)
    columns = 4
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(20, 3.6 * rows), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        for phase, color in (("train", "tab:blue"), ("val", "tab:orange")):
            values = histories.get((phase, metric), {})
            if values:
                epochs = sorted(values)
                axis.plot(epochs, [values[e] for e in epochs], marker="o", markersize=3,
                          linewidth=1.6, label=phase, color=color)
        axis.set_title(metric)
        axis.set_xlabel("epoch")
        axis.set_ylabel("mean batch loss")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)
    figure.suptitle("Sparse MTL YOLO loss progress", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"metrics={len(metrics)} output={args.output}")
    for phase in ("train", "val"):
        epochs = sorted(histories.get((phase, "loss"), {}))
        print(f"{phase}_completed_epochs={epochs}")


if __name__ == "__main__":
    main()
