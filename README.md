# Rare Event Estimation via Iterative Unalignment

**Authors:** Hanming Yang, Daksh Mittal, Jing Dong, Hongseok Namkoong

This repository contains the official code for the paper *Rare Event Estimation via Iterative Unalignment*.

<p align="center">
  <img src="assets/overview.png" alt="Iterative Unalignment Overview" width="100%">
</p>

## Setup

```bash
conda env create -f environment.yml
conda activate rare
```

## Quickstart

Run Iterative Unalignment on GPT-2 with LoRA rank 16 and scaling 32:

```bash
bash methods/scripts/iu.sh
```

Or run directly with `train.py`:

```bash
python methods/train.py "Once upon a time" 10 \
  --event_type token \
  --event_config_json '{"surrogate":"sum","token":"ActionCode","gt_prob":4.97e-9}' \
  --proposal_type IU \
  --model_id gpt2 \
  --steps 300 \
  --eval_steps 500 \
  --batch_size 128 \
  --lr 1e-4 \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --init_lambda 10 \
  --lambda_floor 0.1 \
  --dual_lr 0.01 \
  --dual_optimizer sgd \
  --pop_ess_target 0.005 \
  --ess_target 0.1 \
  --rare_event_ess_min_rate 0.1 \
  --seed 122 \
  --output_path runs/iu_token
```

All ground-truth probabilities (`gt_prob`) are available in the paper's appendix.

## Citation

```bibtex
@article{yang2026iterative,
  title={Rare Event Estimation via Iterative Unalignment},
  author={Yang, Hanming and Mittal, Daksh and Dong, Jing and Namkoong, Hongseok},
  journal={arXiv preprint},
  year={2026}
}
```
