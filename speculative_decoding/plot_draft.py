import matplotlib.pyplot as plt
import numpy as np

steps = [0, 1, 2, 3]

self_draft = {
    "KL(D||V)": [0.0889, 0.0560, 0.1231, 0.2537],
    "KL(V||D)": [0.0248, 0.0247, 0.0320, 0.1435],
    "Loss":     [0.0141, 0.0251, 0.0814, 0.2567],
    "Top1":     [1.0000, 0.9900, 0.9900, 0.9750],
}

cross = {
    "KL(D||V)": [0.1528, 0.1319, 0.2705, 0.5010],
    "KL(V||D)": [0.0597, 0.0647, 0.1118, 0.1493],
    "Loss":     [0.0605, 0.0405, 0.2099, 0.4389],
    "Top1":     [0.9800, 0.9900, 0.9700, 0.9400],
}

xtick_labels = ["0\n(earliest unmask)", "1", "2", "3\n(latest unmask)"]


def plot_one(data, title, out_path):
    plt.rcParams.update({"font.size": 16})
    fig, ax1 = plt.subplots(figsize=(9, 6))
    color_map = {
        "KL(D||V)": "tab:blue",
        "KL(V||D)": "tab:orange",
        "Loss":     "tab:green",
    }
    marker_map = {"KL(D||V)": "o", "KL(V||D)": "s", "Loss": "^"}

    for k in ["KL(D||V)", "KL(V||D)", "Loss"]:
        ax1.plot(steps, data[k], marker=marker_map[k],
                 color=color_map[k], label=k, linewidth=2.5, markersize=10)
    ax1.set_xlabel("Step (unmask order)", fontsize=17)
    ax1.set_xticks(steps)
    ax1.set_xticklabels(xtick_labels, fontsize=15)
    ax1.tick_params(axis="y", labelsize=15)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(steps, data["Top1"], marker="D", color="tab:red",
             label="Top1", linewidth=2.5, markersize=10, linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=15)
    ax2.set_ylim(0.9, 1.01)

    ax1.text(0.0, 1.02, "KL / Loss", transform=ax1.transAxes,
             fontsize=17, ha="center", va="bottom")
    ax2.text(1.0, 1.02, "Top1 accuracy", transform=ax2.transAxes,
             fontsize=17, color="tab:red", ha="center", va="bottom")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=14, ncol=4, frameon=False,
               handletextpad=0.4, columnspacing=1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")


plot_one(self_draft, "Self_draft 8B", "draft_self_8B.png")
plot_one(cross, "Cross 1.7B -> 8B", "draft_cross_1.7B_to_8B.png")
