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
| `activity_profile.py` | Events per 500 ms bin of every training/validation recording: leading and trailing silences that trimming would remove |
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
  their `test_script_genn.py` (tau_theta 420 ms, tau_accumulator 16 ms, beta 1.2, dt 1 ms).
  The author confirmed by email (20 Nov 2026) that these are the values used for
  DVS Gesture in the paper, the same as for the outdoor scenes and not tuned for DVS Gesture.
  (`ncaltech_testing.py` passes tau_theta 16 and tau_accumulator 420 to the NumPy version.)
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
| `delays.csv` | Only with `--delay-layers`: per epoch and delay connection, mean, sd, median and 95th percentile of the delays (timesteps = ms) and the fractions at 0 and at the maximum |
| `checkpoints/` | Weights (`<epoch>-<layer>-weight.npy`) and, with `--delay-layers`, delays (`<epoch>-<layer>-delay.npy`) after every epoch and before training (`initial-...`) |

`--delay-layers output` learns synaptic delays on the hidden -> output Dense
connection (mlGeNN EventProp delay learning, Adam on the delays). Delays start from
U(0, `--init-delay-ms`) (default 50; a constant start would make mlGeNN skip the
delayed adjoint lookup), are clipped to [0, `--max-delay-ms`] (default 150) and use
`--delay-lr` (default 0.1, in timesteps). These defaults are untuned starting points.

Only the output connection is allowed: with delay learning on any kernel connection
GeNN 5.5.0 fails in `_create_backend` with
`vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)`, both for
Toeplitz (`hidden`, run `20261121_020024`) and procedural-kernel connectivity
(`conv1`, run `20261121_020425`). The Dense output connection builds and its delays
change during training (run `20261121_015858`, 2 epochs).

Output delays did not help (4 s EvDownsampling, grad limit 200, regulariser 5e-9,
seed 1, 100 epochs):

| Run | Delays | Best (epoch) | Last 20 | Train last 20 | s/epoch |
|---|---|---|---|---|---|
| `20261120_203626` | none | 118 (93) | 115.0 | 86.6% | 47.8 |
| `20261121_020639` | output, defaults | 113 (72) | 111.2 | 85.9% | 49.8 |

- The gap (3.9 on the last-20 mean) is about the seed noise (about 4/143), so this
  shows no benefit rather than harm.
- Spike counts and adjoint limits of every hidden layer match the baseline to within 1%.
- Delays (ms): mean 25.4 -> 23.7, sd 14.5 -> 24.4, 95th percentile 47.7 -> 69.9, at 0
  1.8% -> 28%, none at 150. The sd was still growing at epoch 99.
- Delays are therefore off by default. Delays inside the network (where they could
  combine features over time) need kernel connectivity and are blocked by the GeNN
  error above.

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

### lambda_V diagnosis (`--lambda-epochs 0 4 9 19`, 20 epochs, seed 1)

| Run | Trial, grad limit | Validation at epoch 19 | conv1 jumps clipped to +/-limit (epochs 4 / 9 / 19) | conv1 limit at epoch 19 |
| --- | --- | --- | --- | --- |
| `dvs_gesture_b32_20261120_164406` | whole, 100 | 59/143 | 99.9% / 99.98% / 100% | 8.2e-13 |
| `dvs_gesture_b32_20261120_170357` | whole, 1000 | 55/143 | 0.02% / 0.02% / 0.03% | 2.4e-5 |
| `dvs_gesture_b32_20261120_172859` | first 2000 ms, 100 | 89/143 | 0.15% / 0.09% / 0.03% | 1.8e-5 |

* With grad limit 100 on whole trials almost every conv1 jump (and, at epoch 4, every
  conv2 jump) is clipped, so lambda_V there is set by the clamp and the clamp
  shrinks each batch. With grad limit 1000, or 2000 ms trials, the clamp almost never binds.
* Many lambda_V values are exactly zero in all runs, also at epoch 0 when the limit is
  still 1.0 (no clipping): 84% of conv1 neurons have lambda_V = 0 throughout the first
  example, and over all batches 64-77% of conv1 and 18-34% of `hidden`
  (neuron, example) pairs. A hidden neuron's lambda_V only jumps at its own spikes,
  so a neuron that does not fire in an example gets no gradient from it; after a
  spike lambda_V decays with tau_mem and reaches float32 zero after long silences.
* A grad limit of 1000 removes the clipping but does not make whole trials learn as
  fast as 2000 ms trials. In mlGeNN the output drive of `avg_var_exp_weight` is divided
  by the trial length, while the `SpikeCount` drive added at each spike is not, so the
  same regulariser strength is about 9x stronger relative to the loss on whole trials.
* Re-running the same seed is not bit-identical on the GPU: validation differed by up
  to 4/143 per epoch from the Stage 1 runs (`144157`, `131943`) over the first 20 epochs.

### Regulariser strength on whole real-time trials (grad limit 1000, seed 1, 20 epochs)

| Run | `--reg-strength` | Validation / train at epoch 19 | conv1 `LambdavLimit` at epoch 19 | Train spikes per neuron (conv1 / conv2 / hidden) |
| --- | --- | --- | --- | --- |
| `dvs_gesture_b32_20261120_170357` | 1e-8 1e-8 | 55/143, 408/934 | 2.4e-5 | 45 / 50 / 95 |
| `dvs_gesture_b32_20261120_174642` | 1e-9 1e-9 | 76/143, 525/934 | 2.4e-6 | 45 / 50 / 96 |
| `dvs_gesture_b32_20261120_181129` | 0 1e-10 | 81/143, 617/934 | 2.4e-8 | 14 / 10 / 72 |
| `dvs_gesture_b32_20261118_133711` (seed 0, earlier EvDownsampling code) | 0 1e-10 | 83/143, 622/934 | 3.1e-8 | 15 / 12 / 75 |

* The limit (proportional to the mean |lambda_V|) scales with the regulariser
  strength, so with 1e-8 or 1e-9 the hidden-layer adjoints on whole trials are
  almost entirely `SpikeCount` drive; without it they are about 1000x smaller.
* With almost no regulariser the run reproduces `133711`, so its better accuracy came
  from the regulariser setting, not from its seed or the earlier EvDownsampling code.
* The same setting over 200 epochs (`dvs_gesture_b32_20261118_154008`) peaked at
  108/143 (epoch 63) while the conv1 limit fell from ~2e-8 (epoch 49) to 9e-12 (epoch 199).

### Activity profile of the recordings (`activity_profile.py`)

`Experiments/dvs_gesture_activity_20261120_191055` (training and validation subjects,
500 ms bins, after denoising):

* There are no silent stretches inside the recordings: with a threshold of 10% of the
  busiest bin, the median leading and trailing silence is 0 ms (95th percentile 0 and
  500 ms) and trimming would keep >= 99.2% of the events in 95% of recordings.
* Recording length: median 6.4 s, 95th percentile 9.4 s; 13% are longer than 8 s and
  3.3% longer than 10 s. The longest training recording is 12.8 s; the 18,454 ms trial
  length comes from one validation recording (subject 21, whose recordings average 9.5 s).
* So on whole-recording trials the silence is the padding after each gesture: a median
  recording is followed by about 12 s of silence, and the loss drive is averaged over it.

### Real-time window (EvDownsampling, seed 1, 100 epochs)

Grad limit and regulariser strength are scaled with the window length from the 2 s run
(grad limit 100, strength 1e-8), so the regulariser keeps the same weight against the
1/T loss drive. Validation is out of 143; "last 20" is the mean of epochs 80-99.

| Run | Trial | Grad limit | Regulariser | Best (epoch) | Last 20 | Train, last 20 | s/epoch |
|---|---|---|---|---|---|---|---|
| `20261120_131943` | first 2 s | 100 | 1e-8 | 113 (83) | 107.4 | 88.8% | |
| `20261120_203626` | first 4 s | 200 | 5e-9 | 118 (93) | 115.0 | 86.6% | 52 |
| `20261120_220835` | first 6 s | 300 | 3e-9 | 118 (97) | 114.5 | 82.0% | 56 |
| `20261120_235241` | first 8 s | 400 | 2.5e-9 | 117 (97) | 113.5 | 79.0% | 60 |
| `20261120_185021` | whole (18.5 s) | 1000 | 0 / 1e-10 | 109 (92) | 106.0 | 81.3% | 35 |

* 4, 6 and 8 s are equal within the seed-to-seed spread (about 4/143) and all beat 2 s
  and the whole recording by about 7-9 examples. 4 s is kept: shortest latency and
  cheapest of the three.
* None of the windowed runs has converged: validation still rises in epochs 90-99 and
  training accuracy is only 79-87%, with a train-validation gap of -0.4 to 6 points.
  The model underfits; it does not overfit.
* With the scaled regulariser the hidden layer sits at its target (95-100 spikes per
  gesture) and the conv layers at 45-62; adjoint limits stay at 4e-6 to 1e-5 throughout.
* Whole recording: the conv1 limit stays at 1e-8 to 3e-8 up to epoch 99 (no collapse
  as in the 200-epoch run). In the epoch-99 trace, 9.9% of conv2's backward jumps hit the
  limit (0.1% at epoch 49), and the median non-zero |lambda_V| is 1e-19 to 1e-14.
  Most of the adjoint has decayed to almost nothing over the 12 s of padding.
* s/epoch follows spike counts, not trial length: the sparse whole-recording run is the
  fastest.
