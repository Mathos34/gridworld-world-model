"""Generate result.png (planning episodes + curves) and architecture.png."""
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
from src.planner import cem_plan  # noqa: E402


def draw_grid(ax, walls, agent, goal, title=""):
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
    ax.add_patch(patches.Circle((agent[1], agent[0]), 0.3, color="#cc2233"))
    for x in np.arange(-0.5, GRID_SIZE, 1):
        ax.axhline(x, color="#ddd", linewidth=0.5)
        ax.axvline(x, color="#ddd", linewidth=0.5)
    ax.set_title(title, fontsize=10)


def render_episode(model, env: GridWorld, max_steps: int = 30, rng=None):
    obs = env.reset()
    traj = [env.agent]
    success = False
    for _ in range(max_steps):
        a = cem_plan(model, obs, env.goal, rng=rng)
        obs, done = env.step(a)
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
    model.load_state_dict(torch.load(runs / "world_model.pt", map_location="cpu"))
    model.eval()

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])

    rng = np.random.default_rng(7)
    shown = 0
    seed = 1000
    while shown < 4 and seed < 1200:
        env = GridWorld(seed=seed)
        walls, traj, goal, success = render_episode(model, env, rng=np.random.default_rng(seed))
        seed += 1
        if not success:
            continue
        ax = fig.add_subplot(gs[0, shown])
        draw_grid(ax, walls, traj[0], goal, title=f"Episode {shown+1}: {len(traj)-1} steps")
        ys = [p[0] for p in traj]
        xs = [p[1] for p in traj]
        ax.plot(xs, ys, color="#cc2233", linewidth=2, alpha=0.7)
        shown += 1

    ax_loss = fig.add_subplot(gs[1, :2])
    ax_loss.plot(metrics["history"]["epoch"], metrics["history"]["pred"], marker="o", color="#1f77b4")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("prediction loss (MSE in latent)")
    ax_loss.set_title("Latent prediction loss")
    ax_loss.grid(alpha=0.3)

    ax_bar = fig.add_subplot(gs[1, 2:])
    names = ["Random baseline", "CEM planner (ours)"]
    vals = [metrics["random_success_rate"] * 100, metrics["planner_success_rate"] * 100]
    colors = ["#999", "#22aa33"]
    bars = ax_bar.bar(names, vals, color=colors)
    ax_bar.set_ylabel("Success rate (%)")
    ax_bar.set_ylim(0, 100)
    ax_bar.set_title(f"Eval on {metrics['eval_episodes']} episodes")
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=11)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.suptitle("Latent world model + CEM planning on a 2D gridworld", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "result.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    def box(x, y, w, h, label, color):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                            edgecolor=color, facecolor="white", linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, color=color)

    box(0.02, 0.55, 0.16, 0.22, "obs_t\n(3 x 8 x 8)", "#444")
    box(0.22, 0.55, 0.12, 0.22, "Encoder\n(MLP)", "#1f77b4")
    box(0.38, 0.55, 0.10, 0.22, "z_t\n(32)", "#1f77b4")
    box(0.52, 0.55, 0.12, 0.22, "Predictor\n(MLP)", "#ff7f0e")
    box(0.68, 0.55, 0.10, 0.22, "ẑ_{t+1}", "#ff7f0e")
    box(0.82, 0.55, 0.16, 0.22, "L2 + VC loss", "#aa2233")

    box(0.02, 0.10, 0.16, 0.22, "obs_{t+1}", "#444")
    box(0.22, 0.10, 0.12, 0.22, "Encoder\n(stop-grad)", "#1f77b4")
    box(0.38, 0.10, 0.10, 0.22, "z_{t+1}", "#1f77b4")
    ax.annotate("", xy=(0.82, 0.66), xytext=(0.78, 0.66), arrowprops=dict(arrowstyle="->", color="#aa2233"))
    ax.annotate("", xy=(0.82, 0.21), xytext=(0.48, 0.21),
                arrowprops=dict(arrowstyle="->", color="#aa2233", connectionstyle="arc3,rad=-0.2"))

    box(0.52, 0.78, 0.12, 0.16, "action a_t\n(0..3)", "#888")
    ax.annotate("", xy=(0.58, 0.77), xytext=(0.58, 0.94), arrowprops=dict(arrowstyle="->", color="#888"))

    ax.text(0.5, 0.97, "Architecture: JEPA-style latent world model + CEM planner",
            ha="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.02, "At inference: CEM searches action sequences in latent space to minimize distance to z_goal.",
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
