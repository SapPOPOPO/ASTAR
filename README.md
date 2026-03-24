# ASTAR

Adversarial Sequence Transformation Augmentation for Recommendation.

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Baseline ASTAR (bugfixed)

Adversarial mask-based augmentation with Phase-1 CL view fix and gradient-stable
augmenter updates.

```bash
python src/main.py \
    --model_name ASTAR \
    --data_name Beauty \
    --data_dir /path/to/data/ \
    --output_dir output/
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--warmup_epochs` | 10 | Epochs of recommender-only warmup before adversarial augmenter updates start |
| `--reg_weight` | 0.2 | Mask-rate regularisation weight |
| `--asym_weight` | 0.2 | Mask asymmetry penalty weight |
| `--mask_tau` | 10.0 | Initial Gumbel-Softmax temperature |

### ASTARv2 dual-view transport ablation

Separate ablation path using two independent transport heads to produce two
augmented views, trained with a view-to-view contrastive objective.

```bash
python src/main.py \
    --model_name ASTARv2 \
    --data_name Beauty \
    --data_dir /path/to/data/ \
    --output_dir output/
```

Key options (ASTARv2-specific):

| Argument | Default | Description |
|---|---|---|
| `--warmup_epochs` | 10 | Warmup epochs before augmenter reg updates start |
| `--v2_cl_weight` | 0.2 | View-to-view contrastive loss weight |
| `--transport_reg_weight` | 0.1 | Transport entropy + balance regularisation weight |
| `--transport_K` | 4 | Number of inter-sequence samples for pool construction |

### CoSeRec (original baseline)

```bash
python src/main.py \
    --model_name CoSeRec \
    --data_name Beauty \
    --data_dir /path/to/data/ \
    --output_dir output/
```

## Evaluation only

Add `--do_eval` to load a saved checkpoint and run test evaluation:

```bash
python src/main.py --model_name ASTAR --data_name Beauty --do_eval
```
