# DVS Gesture EventProp baseline

The target library is `genn-team/ml_genn` branch `ml_genn_3` at
`a00d91bf18450716d099e4d1caf400c0aaf40b2a`, with PyGeNN 5.5.0 and Tonic
1.5.0. The manual data and shape checks passed. The first sorted training
parent, `user01_fluorescent/0.npy`, produced 2,965 occupied bins from 4,414
in-window events. The two fused pool shapes were (7, 7, 16) and (3, 3, 8).
The revised network built, loaded and unloaded on CUDA with six populations
and five connections in `Experiments/task12_cuda_compile_gOtug92R`.

## Data

Use the official Tonic training partition. Subjects 1–20 supply 934 training
parents and subjects 21–23 supply 143 validation parents. Each parent provides
one trial. The official test partition is reserved for the final evaluation.

The reference training script applies Tonic denoising at 10,000 µs and spatial
downsampling by 0.25. We use those transforms. For this short-prefix baseline,
the first retained event defines time zero. Keep the interval [0, 200) ms. A
shorter sample has a silent tail. An empty sample remains silent with its label.
Each occupied (millisecond, y, x, numeric polarity) emits one spike at the bin
start. This deliberately discards count magnitude. The mlGeNN adapter converts
the integer bin starts from µs to ms. Each polarity retains its numeric channel.

The 200 ms horizon is a baseline design choice. At the compiler defaults of
1 ms per step and batch size one, a neuron can emit at most 200 spikes per
trial. Two adjacent trials contribute at most 400 entries per neuron to the
500-slot EventProp ring. The active input field has at most 409,600 occupied
bins per trial, within the default one-million-entry SpikeInput allocation.
These bounds concern capacity. Gesture information later in the parent is
outside this experiment.

## Model

| Stage | Shape (height, width, channels) or units |
| --- | --- |
| Event field with a silent border | 32 × 32 × 2 inside 34 × 34 × 2 |
| Valid 5 × 5 convolution, stride 2, LIF | 15 × 15 × 16 |
| Same 3 × 3 convolution, stride 1, LIF | 15 × 15 × 16 |
| Fused sum pooling and same 3 × 3 convolution, LIF | 7 × 7 × 16 pooled, then 7 × 7 × 8 |
| Fused sum pooling and full-field valid convolution, LIF | 3 × 3 × 8 pooled, then 1 × 1 × 256 flattened |
| Dense, leaky integrator voltage readout | 11 units |

The silent border implements the reference's first-layer padding of one on
each side. Both pooling operations use the library's fused connectivity with
`sum=True`. For the first pool on a 15 × 15 map, the GeNN 5.5.0 Toeplitz
implementation can route spikes from an incomplete border window into Conv3.
We accept this boundary behaviour for this baseline. PyTorch's reference
pooling drops that window.

The second pool feeds a valid 3 × 3 convolution with 256 filters on the
entire 3 × 3 × 8 pooled field. Each filter produces exactly one output, with
72 independent weights per unit (18,432 total). This represents the dense
map using a four-dimensional kernel. For this valid convolution the only
output position uses the three complete pooled rows and columns.

The first CUDA build with `AvgPoolDense2D` stopped during backend creation
with a vector range error (index 2, size 2). That connection had a rank-two
kernel alongside rank-four Toeplitz kernels. GeNN 5.5.0's custom weight-update
code indexes the kernel dimensions of merged groups using the archetype's
rank. A mixed-rank merge is the suspected cause. The replacement uses the
public `AvgPoolConv2D` class and passed the subsequent CUDA build.

Hidden neurons, exponential synapses and normal weight initializers use their
constructor defaults. The intended training objective is the official
`EventPropCompiler` with `example_timesteps=200`,
`losses="sparse_categorical_crossentropy"` and the voltage readout
`avg_var_exp_weight`. The DVS reference uses surrogate gradients, a spiking
output and count-MSE. Its geometry inspired this network.

## Build and load on CUDA

Run from the ConvNet worktree with the new environment. Pass a fresh path
whose parent exists. GeNN writes its generated code there.

```bash
./.venv/bin/python -E -s -B dvs_gesture/model.py --compile-in /path/to/new/build
```

The check uses the stock compiler's default Adam for trainable weights.
It reports the loaded backend and network size, then releases the model.
The success line is `CUDA build, load and unload: OK`.

## Check one weight update

`train.py DATA_ROOT --run-dir NEW_DIRECTORY` calls public `train_validate`
for one epoch on the first two sorted training parents, followed by the first
validation parent. The batch size is one, the GeNN seed is zero and shuffling
is off. EventProp replays the first parent's backward pass during the second
batch. The stock loop schedules one Adam update and ends without a separate
training batch to process the second parent's gradient.

The run prints online training accuracy and validation accuracy at fixed
weights separately. The smoke check requires finite readouts and weights,
the expected sample counts, and a change in at least one saved weight across
the five connections. Initial and end-of-epoch weight files have different
keys in the fresh run directory. These files contain no Adam moments.
Accuracy on this subset is only a smoke metric.

The user-run smoke in `Experiments/task12_train_smoke_Al03UcRh` passed with
0/2 online and 0/1 fixed-weight validation correct. The changed weight counts
from input to output were 800, 2,298, 1,137, 6,032 and 748.

## Two-epoch class pilot

Add `--pilot` to the smoke command and use a new run directory. A NumPy
generator seeded with zero selects two training parents and one validation
parent per class without replacement from their respective subject partitions.
The log records parent paths and labels. The pilot starts from fresh weights.
Both epochs run in one compiled context to retain Adam state. Epoch two moves
the last training parent to the front. With 22 forward training batches per
epoch the stock compiler schedules 21 learning updates, for 42 across the
pilot. The two epochs omit different final parents from backward coverage.

Each epoch writes five weight arrays and checks their finiteness against the
preceding snapshot. `selection.json` records the online and fixed-weight
correct counts, denominators, final training parent and checkpoint key for
each completed epoch. The best key uses the highest validation correct count,
with the earliest epoch winning a tie. The latest key always names the last
completed epoch. Eleven validation parents make this a checkpoint-selection
exercise; full validation and independent testing are separate stages.

In `Experiments/task12_class_pilot_YVyInPfZ`, epochs one and two scored
1/22 and 2/22 online and both scored 1/11 on the fixed validation subset.
The recorded best key was `epoch1`, and the latest was `epoch2`. Each of
the five connections changed in both epochs.

Use `--validate-pilot PILOT_RUN` in a new process and output directory to
verify the selected weights. The command recreates the pilot validation
subset using the same seeded sampler, loads the selected files through
`Network.load`, and compares all five device weight arrays to disk. It
predicts with the public CUDA inference compiler and resets synaptic input
between parents, following the official convolution example. The file
`validation.npz` records paths, labels, scores and predictions. The pilot
saved aggregate accuracy, so only the correct count can be compared across
the two runs.

In `Experiments/task12_pilot_reload_2YSZU3uo`, all five reloaded device
weight arrays matched the selected epoch-one files. Frozen inference again
scored 1/11. The saved predictions included six distinct classes.

## One full development epoch

Use `--full-epoch` with a new output directory. The model starts from
fresh weights. The 934 official training-partition parents from subjects
1–20 are permuted once using NumPy seed zero. All 143 validation parents
from subjects 21–23 are evaluated in sorted path order. The log records the
paths in presentation order. The official test split is reserved.

At batch size one, the training metric covers 934 forward trials and the
stock EventProp loop schedules 933 learning updates. The last training parent
is identified in `selection.json`. The single completed epoch supplies both
best and latest keys. `epoch1-validation.npz` saves the 143 parent paths,
labels, 11-class readouts, predictions and checkpoint key from the fixed-weight
validation pass for comparison after reload.

The first full run, `Experiments/task12_full_epoch1_bAuBsRTS`, recorded
243/934 online training correct and 47/143 fixed-weight validation correct.
Its selected and latest checkpoint keys are both `epoch1`.

Pass `--validate-full TRAINING_RUN` with a fresh `--run-dir` to compare
independent inference to the selected epoch's validation archive.
The check verifies the selected key, ordered parent paths and labels,
all five device weight arrays, and each predicted class. It reports exact
score matches plus maximum and mean absolute differences across all 1,573
scores. The success line refers to predicted-class agreement. The new
`validation.npz` retains the inferred scores even if a prediction differs.

In `Experiments/task12_full_reload_usig5P7J/reload.log`, the independent
CUDA inference run matched all five epoch-one weight arrays and reproduced
all 143 predicted classes. Both runs classified 47/143 validation parents
correctly (32.87%). Of the 1,573 class scores, 406 matched exactly. The
maximum absolute difference was 1.90734863e-05 and the mean absolute
difference was 8.13572436e-07. The official test split remains unevaluated.

## Multi-epoch training on the full split

Use a fresh output directory for each command. A ten-epoch run follows the
epoch count of the pinned mlGeNN convolutional MNIST example. This is a
chosen training budget for DVS Gesture. Each parent supplies the same
200 ms trial used in the one-epoch baseline.

```bash
./.venv/bin/python -E -s -B dvs_gesture/train.py "$data_root" --run-dir "$job/train" --full-epochs 10
./.venv/bin/python -E -s -B dvs_gesture/train.py "$data_root" --run-dir "$job/reload" --validate-full "$job/train"
```

Training starts from fresh weights and keeps the same compiled model and
Adam state across all epochs. Each epoch uses all 934 training parents once.
After the initial seed-zero shuffle, the order rotates right by one place
per epoch so the preceding epoch's final parent contributes a gradient in
the next epoch. The stock schedule gives 933 learning updates per epoch.
All 143 validation parents are scored at fixed weights after each epoch.

Each completed epoch has its own checkpoint and validation archive.
`selection.json` records the planned epoch count, completed metrics, latest
checkpoint and best checkpoint. The highest validation correct count wins,
with the earliest epoch breaking ties. Run the second command after training
succeeds to verify the selected checkpoint's five weight arrays and all
143 predictions in a separate process.

In `Experiments/task12_full_10epochs_12nNLkyG`, online training accuracy
rose from 223/934 in epoch 1 to 446/934 in epoch 10. Fixed-weight validation
rose from 40/143 to 50/143 (34.97%). Epochs 9 and 10 tied, so `epoch9`
was selected and `epoch10` was the latest checkpoint. The five initial
weight files in this fresh run have different SHA-256 hashes from their
counterparts in the earlier one-epoch run.

Independent inference reproduced all five selected weight arrays and all
143 validation predictions. Of the 1,573 class scores, 431 matched exactly.
The maximum absolute difference was 0.227964401 and the mean was
0.00103723951. The cause of the score differences is undetermined.

## Sources

- DVS reference in the workspace, `deep-snn-deployment-on-spinnaker2-single-chip-using-nir/PTQ/Training/PTQ.py`, dataset transforms and model definition.
- [Official EventProp convolution example](https://github.com/genn-team/ml_genn/blob/a00d91bf18450716d099e4d1caf400c0aaf40b2a/examples/event_prop/latency_mnist_conv.py).
- [GeNN Toeplitz pooling and convolution](https://github.com/genn-team/genn/blob/5.5.0/include/genn/genn/initToeplitzConnectivitySnippet.h), `AvgPoolConv2D`.
