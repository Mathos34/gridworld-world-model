"""Generate result.png (planning episodes + curves + bar plot) and architecture.png."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import GRID_SIZE, GridWorld  # noqa: E402
from src.model import WorldModel  # noqa: E402
from src.planner import plan_action  # noqa: E402


def draw_grid(ax, walls, traj, goal, title=""):
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if walls[r, c]:
                ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#222"))
    ax.add_patch(patches.Circle((goal[1], goal[0]), 0.35, color="#22aa33"))
    if traj:
        ax.add_patch(patches.Circle((traj[0][1], traj[0][0]), 0.3, color="#cc2233"))
        ys = [p[0] for p in traj]
        xs = [p[1] for p in traj]
        ax.plot(xs, ys, color="#cc2233", linewidth=2, alpha=0.7)
    for x in np.arange(-0.5, GRID_SIZE, 1):
        ax.axhline(x, color="#ddd", linewidth=0.5)
        ax.axvline(x, color="#ddd", linewidth=0.5)
    ax.set_title(title, fontsize=10)


def render_episode(model, env: GridWorld, max_steps: int = 50):
    env.reset()
    traj = [env.agent]
    success = False
    for _ in range(max_steps):
        a = plan_action(model, env.agent, env.walls, env.goal)
        _, done = env.step(a)
        traj.append(env.agent)
        if done:
            success = True
            break
    return env.walls, traj, env.goal, success


def main():
    out_dir = ROOT / "assets"
    out_dir.mkdir(exist_ok=True)
    runs = ROOT / "runs"

    with open(runs / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    model = WorldModel()
    model.load_state_dict(torch.load(runs / "world_model.pt", map_location="cpu", weights_only=True))
    model.eval()

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])

    shown = 0
    seed = 1500
    while shown < 4 and seed < 1900:
        env = GridWorld(seed=seed)
        walls, traj, goal, success = render_episode(model, env)
        seed += 1
        if not success:
            continue
        ax = fig.add_subplot(gs[0, shown])
        draw_grid(ax, walls, traj, goal, title=f"Episode {shown+1}: {len(traj)-1} steps")
        shown += 1

    ax_loss = fig.add_subplot(gs[1, :2])
    h = metrics["history"]
    ax_loss.plot(h["epoch"], h["pred"], marker="o", color="#1f77b4", label="latent prediction (MSE)")
    ax_loss.plot(h["epoch"], h["pos"], marker="s", color="#d62728", label="position decode (CE)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training curves")
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_bar = fig.add_subplot(gs[1, 2:])
    names = ["Random", "Learned-WM\nBFS planner", "Oracle BFS\n(true dynamics)"]
    vals = [metrics["random_success_rate"] * 100,
            metrics["planner_success_rate"] * 100,
            metrics["oracle_bfs_success_rate"] * 100]
    colors = ["#999", "#22aa33", "#1f77b4"]
    bars = ax_bar.bar(names, vals, color=colors)
    ax_bar.set_ylabel("Success rate (%)")
    ax_bar.set_ylim(0, 105)
    ax_bar.set_title(f"Success on {metrics['eval_episodes']} fresh episodes")
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=11)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.suptitle("Latent world model + BFS planning on a 2D gridworld", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "result.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")

    def box(x, y, w, h, label, color):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                                            edgecolor=color, facecolor="white", linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color=color)

    box(0.02, 0.55, 0.14, 0.22, "obs_t\n(3 x 8 x 8)", "#444")
    box(0.18, 0.55, 0.12, 0.22, "CNN\nencoder", "#1f77b4")
    box(0.32, 0.55, 0.10, 0.22, "z_t\n(128)", "#1f77b4")
    box(0.44, 0.55, 0.12, 0.22, "Predictor\nMLP", "#ff7f0e")
    box(0.58, 0.55, 0.10, 0.22, "z'_{t+1}", "#ff7f0e")
    box(0.70, 0.55, 0.12, 0.22, "Position\nhead", "#2ca02c")
    box(0.84, 0.55, 0.14, 0.22, "next cell\n(0..63)", "#2ca02c")

    box(0.02, 0.10, 0.14, 0.22, "obs_{t+1}", "#444")
    box(0.18, 0.10, 0.12, 0.22, "Encoder\n(stop-grad)", "#1f77b4")
    box(0.32, 0.10, 0.10, 0.22, "z_{t+1}", "#1f77b4")
    ax.annotate("", xy=(0.58, 0.21), xytext=(0.42, 0.21),
                arrowprops=dict(arrowstyle="->", color="#aa2233", connectionstyle="arc3,rad=-0.2"))
    ax.text(0.50, 0.32, "MSE", ha="center", color="#aa2233", fontsize=10)

    box(0.44, 0.84, 0.12, 0.12, "action a_t", "#888")
    ax.annotate("", xy=(0.50, 0.77), xytext=(0.50, 0.84), arrowprops=dict(arrowstyle="->", color="#888"))

    ax.text(0.5, 0.97, "Architecture: CNN encoder + latent transition predictor + grounded position head",
            ha="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.02, "At inference: build a 64x4 transition table by encoding 64 synthetic obs and advancing each by each action; BFS finds the shortest path.",
            ha="center", fontsize=9, color="#555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_path = out_dir / "architecture.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
