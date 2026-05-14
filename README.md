# gridworld-world-model

A latent world model and CEM planner on a 2D gridworld, in the spirit of JEPA.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green)

![result](assets/result.png)

## What it does

An agent learns to navigate an 8x8 gridworld with random walls and a goal cell, **without ever being told what a good action is**. It explores with random actions, learns the dynamics of its world in a compressed latent space, and at test time plans its way to the goal by imagining action sequences and picking the best one.

## Why it matters

This is a tiny stand-in for the joint-embedding predictive architectures (JEPA) that Yann LeCun and others are pushing as a path to real-world planning agents. The same recipe (encode -> predict in latent -> plan) is what you find in DINO-WM, EB-JEPA and modern model-based RL. Doing it in 8x8 makes the moving parts visible.

## How it works

- **Encoder** (MLP): maps a 3-channel grid (agent, wall, goal) to a 64-dim latent vector.
- **Predictor** (MLP): given (latent, action), predicts the next latent. Trained with L2 to a stop-gradient encoder of the next observation, plus VICReg-style variance and covariance regularizers to prevent latent collapse.
- **Position head** (MLP): a small auxiliary decoder that recovers the agent cell from the latent. This grounds the latent geometry; pure JEPA-style L2-to-goal in latent space is too unconstrained on this problem.
- **CEM planner**: at each step, sample 100 horizon-15 action sequences from a per-step categorical distribution, score them by summed Manhattan distance between the predicted-cell rollout and the goal, refit the distribution to the elites, repeat for 5 iterations, then commit the first action (MPC).

## Architecture

![architecture](assets/architecture.png)

## Quickstart

```bash
git clone https://github.com/Mathos34/gridworld-world-model
cd gridworld-world-model
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python train.py
python scripts/make_viz.py
```

About 3 minutes total on a laptop CPU.

## Results

Trained on 10,000 random trajectories of length 20 for 10 epochs (~3 min CPU). Evaluated on 100 fresh episodes with up to 30 environment steps and MPC replanning every step.

| Metric | Value |
|---|---|
| Planner success rate (CEM, MPC, horizon 15) | **57%** |
| Random-action baseline | 13% |
| Improvement over random | 4.4x |
| Final latent prediction loss (L2) | 0.055 |
| Final position-decode cross-entropy | 0.009 |

The model learns the latent dynamics cleanly (position decode is near-perfect, CE 0.009) but the JEPA-style L2 latent objective alone does not produce a planner-friendly geometry on this problem; the auxiliary position head is what makes planning work. With more capacity (CNN encoder, larger latent, longer training) the success rate can be pushed higher; the choices here favor a 3-minute CPU budget.

## References

- Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination*, ICLR 2020.
- Sobal et al., *Learning World Models with Self-Supervised Visual Pretraining* (DINO-WM), 2024.
- Bardes et al., *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*, ICLR 2022.
- Rubinstein & Kroese, *The Cross-Entropy Method*, 2004.

## About

Built by Mathis Lacombe, AI Maker at the [Intelligence Lab](https://www.ece.fr/intelligence-lab/), ECE Paris.
[LinkedIn](https://www.linkedin.com/in/mathis-lacombe34/) · [Hugging Face](https://huggingface.co/Mathos34400)
