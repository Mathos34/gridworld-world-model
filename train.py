"""Train the latent world model and evaluate the CEM planner."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import TrajectoryDataset
from src.losses import world_model_loss
from src.model import WorldModel
from src.planner import evaluate_planner, evaluate_random

SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=10_000)
    parser.add_argument("--traj-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="runs")
    args = parser.parse_args()

    set_seed(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building dataset: {args.n_traj} trajectories x {args.traj_len} steps")
    t0 = time.time()
    dataset = TrajectoryDataset(n_traj=args.n_traj, traj_len=args.traj_len, seed=SEED)
    print(f"  dataset size = {len(dataset)} transitions, built in {time.time()-t0:.1f}s")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device("cpu")
    model = WorldModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: dict[str, list[float]] = {"loss": [], "pred": [], "var": [], "cov": [], "pos": [], "epoch": []}
    print(f"Training for {args.epochs} epochs on device={device}")
    for epoch in range(args.epochs):
        model.train()
        running = {"loss": 0.0, "pred": 0.0, "var": 0.0, "cov": 0.0, "pos": 0.0, "n": 0}
        for obs_t, act, obs_tp1, pos_t, pos_tp1 in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            obs_t = obs_t.to(device)
            act = act.to(device)
            obs_tp1 = obs_tp1.to(device)
            pos_t = pos_t.to(device)
            pos_tp1 = pos_tp1.to(device)
            z_t, z_pred, z_target, pos_logits_t, pos_logits_tp1 = model(obs_t, act, obs_tp1)
            loss, parts = world_model_loss(z_t, z_pred, z_target,
                                           pos_logits_t, pos_logits_tp1,
                                           pos_t, pos_tp1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = obs_t.shape[0]
            running["loss"] += loss.item() * bs
            running["pred"] += parts["pred"] * bs
            running["var"] += parts["var"] * bs
            running["cov"] += parts["cov"] * bs
            running["pos"] += parts["pos"] * bs
            running["n"] += bs
        for k in ("loss", "pred", "var", "cov", "pos"):
            history[k].append(running[k] / running["n"])
        history["epoch"].append(epoch + 1)
        print(f"  epoch {epoch+1}: loss={history['loss'][-1]:.4f} pred={history['pred'][-1]:.4f} "
              f"var={history['var'][-1]:.4f} cov={history['cov'][-1]:.4f} pos={history['pos'][-1]:.4f}")

    ckpt = out_dir / "world_model.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved checkpoint to {ckpt}")

    print(f"Evaluating CEM planner on {args.eval_episodes} episodes...")
    model.eval()
    t_eval = time.time()
    plan_success, _ = evaluate_planner(model, n_episodes=args.eval_episodes, max_steps=30,
                                       horizon=15, n_candidates=100, n_elite=10, n_iter=5,
                                       seed=1234)
    rand_success = evaluate_random(n_episodes=args.eval_episodes, max_steps=30, seed=1234)
    print(f"  CEM planner success rate    : {plan_success*100:.1f}%")
    print(f"  Random baseline success rate: {rand_success*100:.1f}%")
    print(f"  Eval took {time.time()-t_eval:.1f}s")

    metrics = {
        "n_traj": args.n_traj,
        "traj_len": args.traj_len,
        "epochs": args.epochs,
        "final_loss": history["loss"][-1],
        "final_pred_loss": history["pred"][-1],
        "final_pos_loss": history["pos"][-1],
        "planner_success_rate": plan_success,
        "random_success_rate": rand_success,
        "history": history,
        "eval_episodes": args.eval_episodes,
    }
    metrics_path = str(out_dir / "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
