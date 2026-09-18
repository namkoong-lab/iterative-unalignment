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

Run an Iterative Unalignment training run on GPT-2:

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
  --steps 500 \
  --eval_steps 500 \
  --batch_size 128 \
  --lr 1e-4 \
  --seed 122 \
  --output_path runs/iu_token
```

## Citation

```bibtex
@article{yang2026iterative,
  title={Rare Event Estimation via Iterative Unalignment},
  author={Yang, Hanming and Mittal, Daksh and Dong, Jing and Namkoong, Hongseok},
  journal={arXiv preprint},
  year={2026}
}
```
