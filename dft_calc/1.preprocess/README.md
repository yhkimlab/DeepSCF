# DeepSCF preprocessing

This directory preserves the preprocessing pipeline used to turn completed
SIESTA calculations into DeepSCF datasets. Run the commands from the DeepSCF
project root.

## Complete example: SCF to non-SCF

This example regenerates the input for the included `examples/0001` structure,
predicts its density with the pretrained Original model, and uses that density
in a single-shot non-SCF calculation. It does not require downloaded HDF5
datasets. The example inputs include `STRUCT.fdf`, `RUN.fdf`, and the required
pseudopotentials; `0002` and `0003` are additional examples.

First install the Python dependencies, run `bash setup.sh`, and build the
bundled SIESTA as described in the [project README](../../README.md#bundled-siesta-source).
Run the following commands in the same Bash session, starting in the DeepSCF
project root. Use a new output directory for each run; the `mkdir` command
below deliberately fails if this example directory already exists.

### 1. Run the reference SCF calculation

```bash
PROJECT_ROOT="$PWD"
SIESTA_BIN="$PROJECT_ROOT/src/siesta-4.1-b4/Obj/siesta"
WORKDIR=results/reproduce/0001

test -x "$SIESTA_BIN"
test -s pretrained/original/model.pt
mkdir -p "$(dirname "$WORKDIR")"
mkdir "$WORKDIR"
mkdir "$WORKDIR/scf"
cp -R dft_calc/1.preprocess/examples/0001 "$WORKDIR/scf/"

(
  cd "$PROJECT_ROOT/$WORKDIR/scf/0001"
  "$SIESTA_BIN" < RUN.fdf > stdout.txt
  for grid in Molecular.RHO Molecular.IOCH rhoatom rhoover; do
    test -s "$grid"
  done
)
```

Confirm that `results/reproduce/0001/scf/0001/stdout.txt` reports `Job completed`
and SCF convergence. The four files checked above are the target and three
input grids required by step 2. Always use the bundled executable: the
DeepSCF feature outputs are not provided by every SIESTA installation.

### 2. Convert and scale the grids

```bash
python dft_calc/1.preprocess/1.convert.py "$WORKDIR/scf" "$WORKDIR/raw"
python dft_calc/1.preprocess/3.collect.py \
  "$WORKDIR/raw" \
  dft_calc/1.preprocess/metadata/original/scale.txt \
  "$WORKDIR/dataset"
```

The conversion produces `raw/{target,feature1,feature2,feature3}/0001.h5`.
Collection produces `dataset/0001.h5`, with `target` shaped `[80,80,80]` and
`feature` shaped `[3,80,80,80]`. The target remains unscaled.

This example reuses a pretrained model, so it uses that model's existing scale
file and skips `2.scale.py`. Calculating new statistics from this single
structure would change the model's inputs. For a newly trained model, use
the separate [pipeline](#pipeline) below to calculate statistics from the
corresponding training dataset.

### 3. Predict the density and measure its error

```bash
python predict.py --config-name=train/original \
  gpu=0 \
  target.path="$WORKDIR/dataset/0001.h5" \
  hydra.job.chdir=true \
  hydra.run.dir="$PROJECT_ROOT/$WORKDIR/predict"

python evaluate.py --config-name=examples/original \
  gpu=0 \
  target.path="$WORKDIR/dataset/0001.h5" \
  hydra.job.chdir=true \
  hydra.run.dir="$PROJECT_ROOT/$WORKDIR/evaluate"
```

Prediction writes `predict/0001.RHO` and `predict/log.txt` under `WORKDIR`.
Evaluation writes per-structure density errors to `evaluate/log.txt` and the
percentage error to `evaluate/summary.txt`. Both commands load
`pretrained/original/model.pt` with its matching `mean3` and `std3`.
They use the first visible GPU if CUDA is available, otherwise the CPU.

The provided prediction configuration matches this example's 20 Å cubic cell
and `80 × 80 × 80` mesh. When using a different structure, match the
prediction configuration and the SIESTA calculation's cell and mesh.
`target.path` and `load.model` are interpreted relative to the project root.

### 4. Use the predicted density in non-SCF

```bash
mkdir "$WORKDIR/non_scf"
cp dft_calc/2.evaluation/input/RUN.fdf "$WORKDIR/non_scf/"
cp "$WORKDIR/scf/0001/"*.psf "$WORKDIR/non_scf/"
cp "$WORKDIR/scf/0001/STRUCT.fdf" "$WORKDIR/non_scf/STRUCT.fdf"
cp "$WORKDIR/predict/0001.RHO" "$WORKDIR/non_scf/In.RHO"

(
  cd "$PROJECT_ROOT/$WORKDIR/non_scf"
  "$SIESTA_BIN" < RUN.fdf > stdout.txt
  test -s Molecular.EIG
)
```

The non-SCF directory uses the same structure and pseudopotentials as the SCF
calculation, together with the supplied non-SCF `RUN.fdf`. Its settings are
`MaxSCFIterations 1`, `SCF.MustConverge .false.`, `SCF.Mix.First .false.`,
and `SCF.Read.Charge .true.`. The cell and mesh match `In.RHO`.

Confirm `Job completed` and one `scf:` iteration in `non_scf/stdout.txt`.
`SCF_NOT_CONV` is expected for this intentionally one-step calculation.
The output contains the total energy and Fermi energy; `Molecular.EIG`
contains the eigenvalues. Successful completion demonstrates the path from
the included structure through a predicted density to non-SCF. Reproducing
all paper experiments also requires regenerating their full datasets and
running their corresponding training/evaluation configurations.

## Pipeline

```bash
python dft_calc/1.preprocess/1.convert.py CALCULATIONS RAW_DATA
python dft_calc/1.preprocess/2.scale.py RAW_DATA scale.txt
python dft_calc/1.preprocess/3.collect.py RAW_DATA scale.txt OUTPUT_DATASET
```

`CALCULATIONS` must contain one directory per case. Each case must contain:

- `Molecular.RHO`: training target
- `Molecular.IOCH`: feature 1
- `rhoatom`: feature 2
- `rhoover`: feature 3

Step 1 writes the four grids to separate HDF5 channels. Step 2 reproduces the
paper pipeline's statistic: the arithmetic mean of every case's individual
mean and standard deviation. It is not a single global, voxel-weighted
statistic. Step 3 leaves `Molecular.RHO` unscaled and applies `mean2` through
`mean4` and `std2` through `std4` to the three input features.

## Reusing a pretrained model

Do not calculate a new scale from an evaluation set. Select the fixed
`metadata/<model>/scale.txt` belonging to the pretrained model and pass
it to step 3. The same evaluation raw data must therefore be collected once
per model scale.

For example, to prepare one evaluation set for the `(R+S)x3` model:

```bash
python dft_calc/1.preprocess/1.convert.py \
  /path/to/evaluation/calculations /tmp/deepscf-evaluation-raw
python dft_calc/1.preprocess/3.collect.py \
  /tmp/deepscf-evaluation-raw \
  dft_calc/1.preprocess/metadata/r_plus_s_x3/scale.txt \
  datasets/evaluation/my_case/r_plus_s_x3
```

The model configuration's `model.mean` and `model.std` are respectively
`mean3` and `std3`, because the residual baseline reconstructed by the model is
the unscaled `rhoatom` grid.

## Preserved training metadata

The five scale files under `metadata/` are tied to these calculations:

| Model directory | Paper name | Original calculation directory |
| --- | --- | --- |
| `original` | Original | `calc_no_aug` |
| `r` | R | `calc_only_rotation` |
| `r_plus_s` | R+S | `calc_aug_1` |
| `r_plus_s_x2` | (R+S)x2 | `calc_aug_2` |
| `r_plus_s_x3` | (R+S)x3 | `calc_aug_3` |

`mean1` and `std1` are retained for full provenance, although step 3 does not
use them.
