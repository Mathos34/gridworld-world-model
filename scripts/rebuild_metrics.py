"""Re-evaluate from existing checkpoint and rebuild runs/metrics.json from training.log."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import WorldModel  # noqa: E402
from src.planner import evaluate_planner, evaluate_random  # noqa: E402

LOG = ROOT / "training.log"
RUNS = ROOT / "runs"
CKPT = RUNS / "world_model.pt"

pattern = re.compile(
    r"epoch (\d+): loss=([\d.]+) pred=([\d.]+) var=([\d.]+) cov=([\d.]+) pos=([\d.]+)"
)
history = {"epoch": [], "loss": [], "pred": [], "var": [], "cov": [], "pos": []}
with open(LOG, encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            history["epoch"].append(int(m.group(1)))
            history["loss"].append(float(m.group(2)))
            history["pred"].append(float(m.group(3)))
            history["var"].append(float(m.group(4)))
            history["cov"].append(float(m.group(5)))
            history["pos"].append(float(m.group(6)))

print(f"Parsed {len(history['epoch'])} epochs from log")

model = WorldModel()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

print("Evaluating CEM planner on 100 episodes...")
t0 = time.time()
plan_success, _ = evaluate_planner(model, n_episodes=100, max_steps=30,
                                   horizon=15, n_candidates=100, n_elite=10, n_iter=5,
                                   seed=1234)
rand_success = evaluate_random(n_episodes=100, max_steps=30, seed=1234)
print(f"  CEM planner success rate    : {plan_success*100:.1f}%")
print(f"  Random baseline success rate: {rand_success*100:.1f}%")
print(f"  Eval took {time.time()-t0:.1f}s")

metrics = {
    "n_traj": 10000,
    "traj_len": 20,
    "epochs": len(history["epoch"]),
    "final_loss": history["loss"][-1] if history["loss"] else None,
    "final_pred_loss": history["pred"][-1] if history["pred"] else None,
    "final_pos_loss": history["pos"][-1] if history["pos"] else None,
    "planner_success_rate": plan_success,
    "random_success_rate": rand_success,
    "history": history,
    "eval_episodes": 100,
}
out_path = str(RUNS / "metrics.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics to {out_path}")
