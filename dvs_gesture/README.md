# DVS Gesture ConvSNN trained with EventProp (mlGeNN 3)

A small convolutional spiking network trained on DVS Gesture with mlGeNN's
`EventPropCompiler`. The layer sizes follow the surrogate-gradient model in
`deep-snn-deployment-on-spinnaker2-single-chip-using-nir` (`PTQ/Training/PTQ.py`);
the training here is EventProp, not surrogate gradients.

Tested against mlGeNN `ml_genn_3` at `a00d91bf18450716d099e4d1caf400c0aaf40b2a`,
PyGeNN 5.5.0, Tonic 1.5.0 and EvDownsampling (`anindyaghosh/EvDownsampling`) at
`1971fb35f37bff4aa6e1d3b3d749e7d3c560fd1a`.

## Files

| File | Purpose |
| --- | --- |
| `data.py` | Load DVS Gesture with Tonic, split by subject, encode each recording as input spikes |
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
  contains at least one event, using mlGeNN's
  `preprocess_tonic_spikes(dt=1, histogram_thresh=1)`.
* `--encoding evdownsampling`: the authors' real-time GeNN implementation
  (`EventDownsampleGeNN` in `event_downsampling_genn.py`), with the parameters of
  their `test_script_genn.py` (tau_theta 420 ms, tau_accumulator 16 ms, beta 1.2).
  It replaces the 4x spatial downsampling. The repository is expected next to this
  one (`Projects/EvDownsampling`); override with `EVDOWNSAMPLING_REPO=...`. Its
  GeNN code is built in `<cache dir>/evdownsampling_genn/`.
* The SpikeInput buffer is `BATCH_SIZE * calc_max_spikes(...)`, as in the mlGeNN examples.

### Checks against the library and the authors' code (commit `74c640c`)

`check_encodings.py` (removed afterwards) compared the first training recording of
each class (11 recordings):

* the previous hand-written occupancy encoder and `preprocess_tonic_spikes` gave
  identical spikes for all 11;
* the previous hand-written EvDownsampling (from Eq. 1 of the paper) and
  `EventDownsampleGeNN` gave 1,807,106 vs 1,807,110 spikes, identical in 10
  recordings and 4 spikes apart in one;
* the authors' NumPy version (`event_based_downsampling.py`) failed on all 22 calls:
  `setting an array element with a sequence` when exactly one pixel of a polarity
  fires in a step, and an int32 casting error with float ms times. With integer us
  times it also multiplies beta by 1000.

## Architecture

Same layer sizes as the reference model (`ml_genn` branch `sirine_dvs_gesture`,
`examples/dvs_gesture.py`), with two library-level substitutions:

* The reference's `Conv2D(..., conv_padding=1)` needs integer padding, which only
  exists on `sirine_dvs_gesture`; `ml_genn_3` accepts only `"valid"`/`"same"`.
  A one-pixel silent input border gives the same result.
* The reference's `AvgPoolDense2D` (3x3x8 pooled -> 256) is replaced by
  `AvgPoolConv2D` with a full-field 3x3 valid convolution, i.e. the same 72x256
  dense weights. A trainable `AvgPoolDense2D` uses a rank-2 Toeplitz kernel, and
  GeNN 5.5.0 fails in `_create_backend` with
  `vector::_M_range_check: __n (which is 2) >= this->size() (which is 2)` when it
  is combined with a rank-4 Toeplitz `AvgPoolConv2D` in one training model; alone
  it builds (`check_build.py`, commit `c277896`).

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
| `adjoint_limits.csv` | Per epoch and hidden layer: EventProp's adaptive adjoint clamp `LambdavLimit`, and the fraction of (neuron, example) pairs whose lambda_V was zero for the whole backward pass (from `LambdavAbsSum`, averaged over batches) |
| `lambda_trace.csv` | Only with `--lambda-epochs`: for the first training example of those epochs, per hidden layer, the clamp used, its forward spikes, the fraction of (timestep, neuron) entries with lambda_V = 0, neurons with lambda_V = 0 throughout, jumps clipped to exactly +/-`LambdavLimit` (total and per forward spike), and median / max abs lambda_V |
| `checkpoints/` | Weights after every epoch (`<epoch>-<layer>-weight.npy`) and before training (`initial-...`) |

`--train-all` trains on all 23 training subjects (1077 recordings) without
validation; choose `--epochs` beforehand from validation runs. `evaluate.py`
then uses the last epoch and only accepts `--test`.

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

Because the limit is divided by the number of timesteps, the same `grad_limit` is
about 9x tighter on whole real-time trials (18,454 steps) than on 2000 ms trials.
EvDownsampling, real time, regulariser 1e-8, grad limit 100, seed 1, 100 epochs:

| Run | Trial | Best validation | Last-20 mean | Last train | conv1 `LambdavLimit` (epochs 9 / 99) |
| --- | --- | --- | --- | --- | --- |
| `dvs_gesture_b32_20261120_144157` | whole recording, 18,454 ms | 95/143 (epoch 58) | 86.5 | 68.4% | 6.3e-11 / 3.7e-28 |
| `dvs_gesture_b32_20261120_131943` | first 2000 ms (`--crop-ms 2000`) | 113/143 (epoch 83) | 107.4 | 88.8% | 1.7e-5 / 1.8e-5 |

On whole trials the conv1 limit shrinks by about 80x every 10 epochs; on 2000 ms
trials all limits stay at 1e-5 to 6e-5.
