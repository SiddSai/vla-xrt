# Prior-art boundary

VLA-XRT is not a claim to be the first VLA red-teaming framework, the first instruction attack, or the first physical-scene attack.

- **RedVLA** motivates trajectory-aware, task-feasible risk-scene synthesis and gradient-free physical risk amplification. VLA-XRT adopts the feasibility-first principle; the v0.1 pilot uses manually specified reset-time scene patches before attempting trajectory-guided optimization. [Project](https://redvla.github.io/) · [arXiv](https://arxiv.org/abs/2604.22591)
- **ERT** motivates offline, context-grounded generation of diverse task instructions. VLA-XRT keeps generation outside the policy-control loop and records the candidate provenance and human review state. [Paper](https://arxiv.org/abs/2411.18676) · [Code](https://github.com/Improbable-AI/embodied-red-teaming)
- **Q-DIG** motivates quality-diversity instruction search. Its code was listed as forthcoming when this project was initialized, so VLA-XRT does not depend on it. A MAP-Elites archive is later work, after a real joint effect has been demonstrated. [Paper](https://arxiv.org/abs/2603.12510)
- **RoboGCG** motivates testing persistent instruction-channel vulnerabilities but is a targeted, white-box action attack. It is a later baseline, not part of the task-preserving pilot. [Paper](https://arxiv.org/abs/2506.03350) · [Code](https://github.com/eliotjones1/robogcg)
- **VLATTACK** motivates alternating updates across visual and textual coordinates. VLA-XRT's existing `alternating_search` is a transparent candidate-bank coordinate-search baseline; it does not copy VLATTACK implementation or use imperceptible image perturbations. [Paper](https://arxiv.org/abs/2310.04655)

The target contribution is a matched, held-out evaluation of **task-feasible instruction–scene interaction** in closed-loop VLA control.
