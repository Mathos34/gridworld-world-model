"""Train the latent world model (CNN encoder + multi-step JEPA loss) and
evaluate the BFS planner over the learned simulator."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import TrajectoryDataset, UniformTransitionDataset
from src.losses import multi_step_world_loss
from src.model import WorldModel, count_params
from src.planner import evaluate_oracle_bfs, evaluate_planner, evaluate_random

SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=8_000)
    parser.add_argument("--traj-len", type=int, default=8)
    parser.add_argument("--data", choices=["traj", "uniform"], default="uniform")
    parser.add_argument("--n-samples", type=int, default=80_000,
                        help="Used when --data uniform")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--out", type=str, default="runs")
    args = parser.parse_args()

    set_seed(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.data == "uniform":
        print(f"Building uniform-transition dataset: {args.n_samples} samples")
        dataset = UniformTransitionDataset(n_samples=args.n_samples, seed=SEED)
    else:
        print(f"Building trajectory dataset: {args.n_traj} traj x {args.traj_len} steps")
        dataset = TrajectoryDataset(n_traj=args.n_traj, traj_len=args.traj_len, seed=SEED)
    print(f"  {len(dataset)} items built in {time.time()-t0:.1f}s")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device("cpu")
    model = WorldModel().to(device)
    print(f"Model: {count_params(model)/1e6:.2f} M params")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: dict[str, list[float]] = {"loss": [], "pred": [], "var": [], "cov": [], "pos": [], "epoch": []}
    print(f"Training for {args.epochs} epochs on {device}")
    for epoch in range(args.epochs):
        model.train()
        running = {"loss": 0.0, "pred": 0.0, "var": 0.0, "cov": 0.0, "pos": 0.0, "n": 0}
        for obs_seq, act_seq, pos_seq in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            obs_seq = obs_seq.to(device)
            act_seq = act_seq.to(device)
            pos_seq = pos_seq.to(device)
            loss, parts = multi_step_world_loss(model, obs_seq, act_seq, pos_seq)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = obs_seq.shape[0]
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

    print(f"Evaluating BFS planner on {args.eval_episodes} episodes...")
    model.eval()
    t_eval = time.time()
    plan_success, steps_taken = evaluate_planner(model, n_episodes=args.eval_episodes,
                                                 max_steps=50, seed=1234)
    rand_success = evaluate_random(n_episodes=args.eval_episodes, max_steps=50, seed=1234)
    oracle_success = evaluate_oracle_bfs(n_episodes=args.eval_episodes, max_steps=50, seed=1234)
    eval_s = time.time() - t_eval
    print(f"  Learned-WM BFS planner : {plan_success*100:.1f}% (avg {np.mean(steps_taken):.1f} steps)")
    print(f"  Oracle BFS (true dyn)  : {oracle_success*100:.1f}%")
    print(f"  Random baseline        : {rand_success*100:.1f}%")
    print(f"  Eval took {eval_s:.1f}s")

    metrics = {
        "n_traj": args.n_traj,
        "traj_len": args.traj_len,
        "epochs": args.epochs,
        "n_params": count_params(model),
        "final_loss": history["loss"][-1],
        "final_pred_loss": history["pred"][-1],
        "final_pos_loss": history["pos"][-1],
        "planner_success_rate": plan_success,
        "random_success_rate": rand_success,
        "oracle_bfs_success_rate": oracle_success,
        "history": history,
        "eval_episodes": args.eval_episodes,
        "avg_steps_to_goal": float(np.mean(steps_taken)),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
