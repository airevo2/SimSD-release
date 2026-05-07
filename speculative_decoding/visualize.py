"""Process visualization for speculative decoding evaluation."""

import json
import os
from collections import defaultdict


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_single_block(run_dir, jsonl_path, block_length):
    """
    Plot single-block evaluation: acceptance dist, top1/kl per sample, running avg.
    Saves to run_dir/plots.png.
    """
    plt = _ensure_matplotlib()
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))

    if not records:
        return

    def _accepted_val(r):
        v = r.get("n_accepted")
        if v is not None:
            return v
        if r.get("draft_eq_target_mode_count") is not None:
            return r["draft_eq_target_mode_count"]
        return 0

    all_accepted = [_accepted_val(r) for r in records]
    all_top1 = []
    all_kl = []
    for r in records:
        all_top1.extend(r.get("top1_match", []))
        all_kl.extend(r.get("per_pos_kl", []))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Acceptance distribution
    ax = axes[0, 0]
    acc_dist = defaultdict(int)
    for a in all_accepted:
        acc_dist[a] += 1
    xs = sorted(acc_dist.keys())
    ys = [acc_dist[x] for x in xs]
    bars = ax.bar(xs, ys, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel("Tokens accepted per block")
    ax.set_ylabel("Count")
    _ppc = records and records[0].get("speculative_branch") == "per_position_compare"
    ax.set_title(
        "draft_id==target_top1 / block" if _ppc else "Acceptance Distribution"
    )
    ax.set_xticks(range(block_length + 1))
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, str(y), ha="center", fontsize=9)

    # 2. Accepted count per sample (over sample index)
    ax = axes[0, 1]
    ax.bar(range(len(all_accepted)), all_accepted, color="coral", alpha=0.7, width=0.8)
    ax.axhline(y=sum(all_accepted) / len(all_accepted), color="red", linestyle="--", label="Avg")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Count" if _ppc else "Accepted")
    ax.set_title("Per-sample count" if _ppc else "Accepted per Sample")
    ax.legend()
    ax.set_ylim(-0.5, block_length + 0.5)

    # 3. Running avg accepted (cumulative)
    ax = axes[1, 0]
    cum_acc = [sum(all_accepted[:i + 1]) / (i + 1) for i in range(len(all_accepted))]
    ax.plot(range(1, len(cum_acc) + 1), cum_acc, "b-", linewidth=2)
    ax.axhline(y=sum(all_accepted) / len(all_accepted), color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Sample index (cumulative)")
    ax.set_ylabel("Running avg (count/block)" if _ppc else "Running avg accepted")
    ax.set_title(
        "Running avg draft==target_top1" if _ppc else "Process: Running Average Accepted"
    )
    ax.set_ylim(0, block_length + 0.2)
    ax.grid(True, alpha=0.3)

    # 4. Top1 match and KL (per-position, across samples)
    ax = axes[1, 1]
    if all_top1 and all_kl:
        n_pos = block_length
        top1_by_pos = [0.0] * n_pos
        kl_by_pos = [0.0] * n_pos
        cnt = [0] * n_pos
        for i, t in enumerate(all_top1):
            pos = i % n_pos
            top1_by_pos[pos] += t
            cnt[pos] += 1
        for i, k in enumerate(all_kl):
            pos = i % n_pos
            kl_by_pos[pos] += k
        for p in range(n_pos):
            if cnt[p]:
                top1_by_pos[p] /= cnt[p]
                kl_by_pos[p] /= cnt[p]
        ax2 = ax.twinx()
        l1, = ax.plot(range(n_pos), top1_by_pos, "o-", color="green", label="Top1 match")
        l2, = ax2.plot(range(n_pos), kl_by_pos, "s-", color="orange", label="KL(q||p)")
        ax.set_xlabel("Position in block")
        ax.set_ylabel("Top1 match rate", color="green")
        ax2.set_ylabel("Avg KL", color="orange")
        ax.set_ylim(0, 1.05)
        ax.legend(handles=[l1, l2], loc="upper right")
    else:
        ax.text(0.5, 0.5, "No per-position data", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Top1 Match & KL by Position")

    tag = os.path.basename(run_dir)
    fig.suptitle(f"Single-Block Spec Decode: {tag}\nbl={block_length}", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(run_dir, "plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_multi_block(run_dir, jsonl_path, block_length):
    """
    Plot multi-block evaluation: accept rate dist, per-block accepted, process over samples.
    Saves to run_dir/plots.png.
    """
    plt = _ensure_matplotlib()
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))

    if not records:
        return

    all_accept_rates = [r["accept_rate"] for r in records]
    all_per_block = []
    for r in records:
        all_per_block.extend(r.get("per_block_accepted", []))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Accept rate distribution (histogram)
    ax = axes[0, 0]
    ax.hist(all_accept_rates, bins=15, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.axvline(x=sum(all_accept_rates) / len(all_accept_rates), color="red", linestyle="--", label="Mean")
    ax.set_xlabel("Accept rate")
    ax.set_ylabel("Count")
    ax.set_title("Accept Rate Distribution")
    ax.legend()

    # 2. Per-block accepted distribution
    ax = axes[0, 1]
    acc_dist = defaultdict(int)
    for a in all_per_block:
        acc_dist[a] += 1
    xs = sorted(acc_dist.keys())
    ys = [acc_dist[x] for x in xs]
    ax.bar(xs, ys, color="coral", edgecolor="darkred", alpha=0.8)
    ax.set_xlabel("Accepted per block")
    ax.set_ylabel("Count")
    ax.set_title("Per-Block Acceptance Distribution")

    # 3. Accept rate over sample index (process)
    ax = axes[1, 0]
    ax.plot(range(len(all_accept_rates)), all_accept_rates, "b-", alpha=0.6)
    cum = [sum(all_accept_rates[:i + 1]) / (i + 1) for i in range(len(all_accept_rates))]
    ax.plot(range(len(cum)), cum, "r-", linewidth=2, label="Running avg")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Accept rate")
    ax.set_title("Process: Accept Rate per Sample")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 4. Gen length vs accept rate scatter
    ax = axes[1, 1]
    gen_lens = [r.get("gen_len", 0) for r in records]
    ax.scatter(gen_lens, all_accept_rates, alpha=0.6, c="green", s=30)
    ax.set_xlabel("Generated length")
    ax.set_ylabel("Accept rate")
    ax.set_title("Gen Length vs Accept Rate")

    tag = os.path.basename(run_dir)
    fig.suptitle(f"Multi-Block Spec Decode: {tag}\nbl={block_length}", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(run_dir, "plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
