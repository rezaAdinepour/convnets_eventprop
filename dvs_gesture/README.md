# DVS Gesture ConvSNN trained with EventProp (mlGeNN 3)

A small convolutional spiking network trained on DVS Gesture with mlGeNN's
`EventPropCompiler`. The layer sizes follow the surrogate-gradient model in
`deep-snn-deployment-on-spinnaker2-single-chip-using-nir` (`PTQ/Training/PTQ.py`);
the training here is EventProp, not surrogate gradients.

Tested against mlGeNN `ml_genn_3` at `a00d91bf18450716d099e4d1caf400c0aaf40b2a`,
PyGeNN 5.5.0 and Tonic 1.5.0.

## Files

| File | Purpose |
| --- | --- |
| `data.py` | Load DVS Gesture with Tonic, split by subject, encode each recording as input spikes |
| `evdownsampling.py` | EvDownsampling encoder (Ghosh, Nowotny and Knight, Eq. 1) |
| `model.py` | Network definition and the training / inference compilers |
| `train.py` | Training loop with per-epoch CSV logs and checkpoints |
| `evaluate.py` | Reload the best epoch and evaluate validation (or, once, the test set) |
| `plot_training.py` | Plot validation accuracy and training loss of one or more runs |
| `run.sh` | Run `train.py` in a new, timestamped experiment directory |

## Data

* Official train set split by subject: subjects 1-20 for training (934 recordings),
  21-23 for validation (143). The official test set (subjects 24-29, 264) is only
  used by `evaluate.py --test`.
* Denoise (10 ms), downsample 128x128 to 32x32, add a one-pixel silent border (34x34x2).
* Whole recordings at 1 ms bins. By default all trials last as long as the longest
  recording (18,454 ms) and shorter recordings are followed by silence.
  `--trial-ms T` instead compresses (or stretches) every recording linearly into
  exactly T ms before encoding, so each gesture fills the whole trial. Time
  constants (neurons and EvDownsampling) then act on compressed time.
  `--crop-ms T` instead keeps only the first T ms of each recording in real time
  (from its first event, before denoising); recordings shorter than T are followed
  by silence. `--trial-ms` and `--crop-ms` cannot be combined.
* `--encoding occupancy` (default): one spike per pixel, polarity and 1 ms bin that
  contains at least one event. `--encoding evdownsampling`: adaptive-threshold encoder.

## Training setup

Batch size 32, Adam (lr 0.001), sparse categorical cross-entropy on the
`avg_var_exp_weight` readout, LIF hidden neurons (tau_mem 20 ms, exponential
synapses 5 ms), `strict_buffer_checking=True`, `max_spikes=6242`. Training data
is reshuffled every epoch (`shuffle=True`).

With mlGeNN's EventProp the backward pass of each batch runs during the next
batch, so the last (partial) batch of every epoch gets no weight update. The
shuffle means a different set of recordings is affected each epoch.

## Running

```bash
cd dvs_gesture
./run.sh --encoding occupancy --epochs 100
```

Encoded recordings are cached in `Experiments/dvs_gesture_encoding_cache/`
(override with `CACHE_DIR=...`), one file per split, encoding and `--trial-ms` / `--crop-ms`.
The file name contains a hash of the recordings, the preprocessing constants and
the encoder code, so changing any of them creates a new cache file instead of
reusing a stale one. `evaluate.py` uses the same cache.

Each run creates `Experiments/dvs_gesture_b32_<date>_<time>/` with
`git_commit.txt`, `train.log` and a `train/` directory containing:

| File | Contents |
| --- | --- |
| `run_config.json` | Encoding, trial length, buffer sizes, grad limit, seed |
| `metrics.csv` | Per epoch: train and validation correct/total, mean training loss, seconds |
| `spikes.csv` | Per epoch, split and hidden layer: mean and max spikes per neuron per gesture, fraction of silent neurons |
| `adjoint_limits.csv` | Per epoch and hidden layer: EventProp's adaptive adjoint clamp `LambdavLimit` |
| `checkpoints/` | Weights after every epoch (`<epoch>-<layer>-weight.npy`) and before training (`initial-...`) |

Evaluate the best epoch (most correct validation predictions, earliest on a tie):

```bash
python evaluate.py ../../../Experiments/reference_data_verified_vdCGRkGx RUN_DIR/train
```

## Known issue: convolutional weights stop changing

In the earlier 100-epoch runs (batch size 1) conv1-conv3 stopped changing after
epoch 32-35. In a batch-32 run (`Experiments/dvs_gesture_AcMYdMDZ`) the same
happened by epoch 3-4, and `adjoint_limits.csv` showed `LambdavLimit` = 0.0 for
conv1-conv3 from then on (hidden decayed to ~1e-14).

`LambdavLimit` is recomputed after every batch as
`grad_limit * sum|lambda_V| / timesteps / batch / neurons` and the jump of
`lambda_V` at each spike is clipped to +/-`LambdavLimit`. Because the mean is
taken over all 18,454 timesteps of a whole-gesture trial, the limit is very
small; once a layer's clipped jumps are all zero, the next limit is zero too and
that layer (and every layer before it) receives no gradient for the rest of
training. A batch-4 run with `--grad-limit 1000` (`dvs_gesture_mDpU3kNE`) kept
all layers changing for 100 epochs.

At batch size 32, `--grad-limit 1000` only delays this: in a 200-epoch run
(`dvs_gesture_b32_20261118_154008`) the conv1/conv2 limits fell from ~2.5e-8
(epoch 40) to ~9e-12 (epoch 199), conv2 activity fell from ~7 to ~1.5 spikes per
neuron, and validation accuracy dropped from 108/143 (epoch 63) to ~101/143.
Shorter trials (`--trial-ms`) are the current attempt to avoid it.
