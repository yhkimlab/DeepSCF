# Configuration layout

The configuration tree mirrors the current directories under `datasets/`.
Every YAML file is self-contained and does not inherit values from another
configuration.

## Model case names

| Paper case | Project name |
|---|---|
| Original | `original` |
| R | `r` |
| R+S | `r_plus_s` |
| (R+S)×2 | `r_plus_s_x2` |
| (R+S)×3 | `r_plus_s_x3` |

## Tree

```text
configs/
├── train/
│   ├── default.yaml
│   ├── original.yaml
│   ├── r.yaml
│   ├── r_plus_s.yaml
│   ├── r_plus_s_x2.yaml
│   ├── r_plus_s_x3.yaml
│   └── hyperparameter/
├── evaluate/
│   ├── default.yaml
│   ├── examples/
│   ├── rotation/guanine/
│   ├── scaling/guanine/
│   └── sequence/
└── predict/
    ├── default.yaml
    ├── train/
    ├── examples/
    ├── rotation/guanine/
    ├── scaling/guanine/
    └── sequence/
```

Hyperparameter experiments exist only under `configs/train/hyperparameter/`.
Evaluation and prediction contain no architecture or hyperparameter test
configurations.

## Usage

Pass a path relative to the corresponding configuration directory and omit the
`.yaml` extension.

```bash
python train.py --config-name=r_plus_s_x3
python train.py --config-name=hyperparameter/kernel/kernel_3

python evaluate.py --config-name=rotation/guanine/r_plus_s_x3

python predict.py --config-name=train/r_plus_s_x2
python predict.py --config-name=examples/r_plus_s_x3
```

Prediction and evaluation load the bundled final weights from
`pretrained/<model_case>/model.pt`. The matching `data.pkl` in each directory
preserves that model's original train/test split.
