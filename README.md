# VLA-XRT

VLA-XRT is a simulator-first protocol for testing whether *paired* instruction and scene changes expose unsafe Vision-Language-Action (VLA) behavior that neither change exposes alone.

This v0.1 release is intentionally a thin, runnable vertical slice:

- a four-arm matched protocol: clean, instruction-only, scene-only, joint;
- JSON scenario specifications and strict result records;
- safety, feasibility, and semantic-preservation gates;
- deterministic mock rollouts for local development and CI;
- manifests, JSONL outputs, and a simple interaction report.

It does **not** claim to run a real VLA by default. Real policies belong behind explicit adapters so that model versions, camera preprocessing, action chunking, and simulator settings remain auditable.

## Real pilot: OpenVLA in LIBERO

`LiberoOpenVLAAdapter` is the integration contract for a pinned upstream OpenVLA + LIBERO deployment. It leaves the upstream OpenVLA template and image/action preprocessing unchanged, and accepts only the task instruction text. A scene patch is applied after reset but before the first policy observation.

The first physical oracle records the maximum motion of one named non-target object. Patches only move an existing non-target free joint and must pass reset-time feasibility checks. `configs/libero_openvla_pilot.json` is the experiment template; replace its checkpoint and task placeholders only after rendering clean rollouts.

The complete execution protocol is in [docs/MINIMUM_PILOT.md](docs/MINIMUM_PILOT.md), and the deliberate prior-art boundaries are in [docs/PRIOR_ART.md](docs/PRIOR_ART.md).

Instruction generation is offline and reviewable, never part of the policy-control loop:

```bash
python -m pip install -e '.[instruction-generation]'
export OPENAI_API_KEY='...'
vla-xrt generate-bank \
  --task-id selected_libero_task \
  --instruction 'Place the bowl on the plate.' \
  --output candidates/selected_libero_task.json
```

Every candidate begins with `human_approved: false`. Review it, set approval explicitly, and freeze the JSON file before any rollout campaign. Do not commit API keys.

## Quick start

```bash
cd /Users/siddsai/Documents/ChatGPT/Career/vla-xrt
python -m pip install -e .
vla-xrt run --config configs/microbench.json
vla-xrt search --config configs/microbench.json --seed 0
vla-xrt analyze results/runs/<run-id>/episodes.jsonl
pytest -q
```

The mock benchmark has an intentional, deterministic cross-modal failure: a task-preserving precision instruction plus a feasible bystander placement causes a keep-out violation, while the two corresponding unimodal arms remain safe. It exists to prove the protocol, not to make a robotics claim.

## Standard result contract

Each episode is one JSON line with the task, seed, arm, exact instruction and scene IDs, task success, safety cost, safety predicate, feasibility decision, semantic-preservation decision, and a provenance block. Never report an aggregate ASR without retaining these rows.

## Real-policy integration

Implement `EnvironmentAdapter.rollout` in `src/vla_xrt/adapters/base.py`. The adapter must return simulator-derived safety signals and preserve enough metadata to replay the episode. Recommended initial integrations:

1. VLA-Arena task/scenario APIs for scene construction and safety constraints.
2. SmolVLA through pinned LeRobot/LIBERO preprocessing.
3. A second policy, such as OpenVLA or OpenPI, only after the first integration is reproducible.

Use held-out seeds for results. Candidates may be selected on a search split, but the final clean / instruction / scene / joint comparison must be re-run unchanged on unseen seeds.

## Scope and safety

Run the search in simulation by default. Hardware execution needs an independent emergency stop, workspace limits, collision monitoring, and human supervision. This repository is designed to measure and mitigate unsafe behavior, not deploy attack payloads to robots.
