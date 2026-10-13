# Dataset layout

This document describes the current on-disk dataset tree used by the project.

## Model case names

| Paper case | Directory name |
|---|---|
| Original | `original` |
| R | `r` |
| R+S | `r_plus_s` |
| (R+S)×2 | `r_plus_s_x2` |
| (R+S)×3 | `r_plus_s_x3` |

## Tree

```text
datasets/
├── train/
│   ├── original/
│   ├── r/
│   ├── r_plus_s/
│   ├── r_plus_s_x2/
│   └── r_plus_s_x3/
├── evaluation/
│   ├── examples/
│   │   ├── original/
│   │   ├── r_plus_s/
│   │   └── r_plus_s_x3/
│   ├── rotation/guanine/
│   │   ├── original/
│   │   ├── r/
│   │   ├── r_plus_s/
│   │   └── r_plus_s_x3/
│   ├── scaling/guanine/
│   │   ├── original/
│   │   ├── r/
│   │   ├── r_plus_s/
│   │   └── r_plus_s_x3/
│   └── sequence/r_plus_s_x3/
└── splits/
    ├── original.pkl
    ├── r.pkl
    ├── r_plus_s.pkl
    ├── r_plus_s_x2.pkl
    └── r_plus_s_x3.pkl
```

The tree describes the prepared datasets; HDF5 files are excluded from Git.
The [complete reproduction example](../dft_calc/1.preprocess/README.md#complete-example-scf-to-non-scf)
regenerates one model input from the included SIESTA structure and calculation
files. The remaining paper structures are under
`dft_calc/1.preprocess/structures/`; prepare their calculation directories with
the matching structure, SCF input, pseudopotentials, and grid dimensions before
applying the same preprocessing pipeline.

Split indexes preserve the original train/test membership using project-relative
paths, which require the corresponding prepared HDF5 files.
