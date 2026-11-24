import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from ml_genn.callbacks import (Callback, Checkpoint, OptimiserParamSchedule, SpikeRecorder,
                               VarRecorder)
from ml_genn.serialisers import Numpy
from ml_genn.utils.data import calc_max_spikes

from data import (BETA, ENCODINGS, TAU_ACCUMULATOR_MS, TAU_THETA_MS, augment_spikes,
                  augmentation_growth, encode_recordings, load_dataset, max_spikes_per_neuron,
                  split_train_validation)
from model import (BATCH_SIZE, CHANNELS, DELAY_LEARNING_RATE, GRAD_LIMIT, INIT_DELAY_MS,
                   LAYER_NAMES, DOWNSAMPLING, LEARNING_RATE, MAX_DELAY_MS, MAX_SPIKES,
                   REG_STRENGTH, REG_TARGET, SEED, SPIKING_WEIGHT_SD, SYNAPSES, TAU_MEM_MS,
                   TAU_SYN_MS, build_network, compile_for_training, delay_connections)

LR_SCHEDULES = ("constant", "cosine")


def load_config(path, run=None):
    """Read a JSON config of settings named like train.py's options (grad_limit, ...).

    "base" names another config, relative to this one, whose settings this one
    overrides. "runs" maps run names to the settings that each run changes.
    """
    config = json.loads(path.read_text())
    base, runs = config.pop("base", None), config.pop("runs", None)
    if base is not None:
        config = {**load_config(path.parent / base), **config}
    if runs is None:
        if run is not None:
            raise ValueError(f"{path} has no runs, so --run {run} is not possible")
        return config
    if run not in runs:
        raise ValueError(f"{path} needs --run, one of: {', '.join(runs)}")
    return {**config, **runs[run]}


def apply_config(parser, config):
    """Make the config's settings the defaults, so command-line options still override them."""
    actions = {action.dest: action for action in parser._actions if action.option_strings}
    for key, value in config.items():
        action = actions.get(key)
        if action is None or key in ("config", "run", "help"):
            parser.error(f"Unknown config setting {key!r}")
        values = value if isinstance(value, list) else [value]
        if isinstance(action.nargs, int) and action.nargs > 0 and len(values) != action.nargs:
            parser.error(f"Config setting {key!r} needs {action.nargs} values")
        if action.choices is not None and any(v not in action.choices for v in values):
            parser.error(f"Config setting {key!r} must be one of {list(action.choices)}")
    parser.set_defaults(**config)


def lr_factor(epoch, schedule, epochs, warmup_epochs):
    """Learning rate of ``epoch`` relative to the initial one."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    if schedule == "cosine":
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    return 1.0


def lr_schedule(schedule, epochs, warmup_epochs):
    # Rescale each optimiser's current alpha, so weight and delay rates keep their ratio.
    def func(epoch, alpha):
        previous = lr_factor(epoch - 1, schedule, epochs, warmup_epochs) if epoch > 0 else 1.0
        return alpha * lr_factor(epoch, schedule, epochs, warmup_epochs) / previous
    return OptimiserParamSchedule("alpha", func)


class CSVLog:
    """Create a CSV file with a header, then append one row per call."""

    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        with open(filename, "w", newline="") as f:
            csv.DictWriter(f, fieldnames).writeheader()

    def write(self, **row):
        with open(self.filename, "a", newline="") as f:
            csv.DictWriter(f, self.fieldnames).writerow(row)


class AdjointLimitLog(Callback):
    """Log LambdavLimit and the fraction of neurons whose lambda_V stayed zero."""

    def __init__(self, log, hidden_layers):
        self.log = log
        self.hidden_layers = hidden_layers
        self.blocked = []

    def create_state(self, compiled_network, **kwargs):
        self.pops = [compiled_network.neuron_populations[layer] for layer in self.hidden_layers]
        self.zero = np.zeros(len(self.pops))
        self.batches = 0
        return compiled_network

    def on_batch_end(self, compiled_network, batch, metrics):
        # LambdavAbsSum = sum over the previous batch's backward pass of |lambda_V|
        # (reset at batch begin); batch 0 is skipped, as by mlGeNN's limit update.
        if batch == 0:
            return
        for i, pop in enumerate(self.pops):
            abs_sum = pop.vars["LambdavAbsSum"]
            abs_sum.pull_from_device()
            self.zero[i] += np.mean(abs_sum.view == 0)
        self.batches += 1

    def on_epoch_end(self, compiled_network, epoch, metrics):
        # A limit of exactly 0 clamps every later jump to 0, so it never recovers.
        self.blocked = []
        for layer, pop, zero in zip(self.hidden_layers, self.pops, self.zero):
            limit = pop.vars["LambdavLimit"]
            limit.pull_from_device()
            limit = float(np.max(limit.values))
            self.log.write(epoch=epoch, layer=layer.name, limit=limit,
                           zero_fraction=zero / max(self.batches, 1))
            if limit == 0.0:
                self.blocked.append(layer.name)


class BackwardLimitRecorder(Callback):
    """Store each layer's LambdavLimit used during batch 1."""

    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.limits = {}

    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def on_batch_begin(self, compiled_network, batch):
        if batch == 1:
            for layer in self.hidden_layers:
                limit = compiled_network.neuron_populations[layer].vars["LambdavLimit"]
                limit.pull_from_device()
                self.limits[layer.name] = np.max(limit.values)


class DelayLog(Callback):
    """Log the distribution of each learnable-delay connection's delays (timesteps)."""

    def __init__(self, log, connections):
        self.log = log
        self.connections = connections

    def create_state(self, compiled_network, **kwargs):
        self.pops = [compiled_network.connection_populations[c] for c in self.connections]
        return compiled_network

    def on_epoch_end(self, compiled_network, epoch, metrics):
        for connection, pop in zip(self.connections, self.pops):
            delay = pop.vars["delay"]
            delay.pull_from_device()
            d = np.asarray(delay.values, dtype=float).ravel()
            self.log.write(epoch=epoch, layer=connection.target().name, count=d.size,
                           mean=d.mean(), sd=d.std(), p50=np.median(d), p95=np.percentile(d, 95),
                           at_zero=np.mean(d <= 0.5),
                           at_max=np.mean(d >= connection.max_delay_steps - 0.5))


def lambda_recorders(layers):
    # Batch 1 runs the backward pass of batch 0, so lane 0 of batch 1
    # holds lambda_V of the first training example of the epoch.
    return [VarRecorder(layer, "Lambdav", key=("lambda", layer.name),
                        example_filter=[BATCH_SIZE]) for layer in layers]


def log_lambda_trace(log, epoch, data, limits, layers):
    for layer in layers:
        lam = np.abs(data[("lambda", layer.name)][0])  # (timesteps, neurons)
        spikes = int(np.sum(data[("train", layer.name)][0]))
        limit = limits[layer.name]
        clipped = int(np.sum(lam == limit)) if limit > 0 else 0
        nonzero = lam[lam > 0]
        log.write(epoch=epoch, layer=layer.name, limit=float(limit), forward_spikes=spikes,
                  zero_fraction=np.mean(lam == 0),
                  zero_neurons=np.mean(lam.max(axis=0) == 0),
                  clipped_jumps=clipped, clipped_per_spike=clipped / max(spikes, 1),
                  median_nonzero=float(np.median(nonzero)) if nonzero.size else 0.0,
                  max_abs=float(lam.max()))


def spike_recorders(layers, split, num_examples):
    # example_filter excludes the unused batch lanes of the last, partial batch.
    return [SpikeRecorder(layer, key=(split, layer.name), record_counts=True,
                          example_filter=np.arange(num_examples))
            for layer in layers]


def log_spike_counts(log, epoch, split, recordings, layers):
    for layer in layers:
        counts = np.array(recordings[(split, layer.name)])  # (examples, neurons)
        log.write(epoch=epoch, split=split, layer=layer.name,
                  mean_per_neuron=counts.mean(), max_per_neuron=int(counts.max()),
                  silent_fraction=np.mean(counts.sum(axis=0) == 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=Path, help="Directory that contains DVSGesture/")
    parser.add_argument("run_dir", type=Path, help="New directory for this run's outputs")
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON file of settings (see load_config); options given here "
                             "override it")
    parser.add_argument("--run", default=None, help="Run name from the config's runs")
    parser.add_argument("--encoding", choices=ENCODINGS, default="occupancy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad-limit", type=float, default=GRAD_LIMIT,
                        help="EventProp adjoint limit multiplier (default: mlGeNN's 100)")
    parser.add_argument("--reg-target", type=float, default=REG_TARGET,
                        help="SpikeCount target spikes per hidden neuron per gesture")
    parser.add_argument("--reg-strength", type=float, nargs=2, default=REG_STRENGTH,
                        metavar=("BELOW", "ABOVE"),
                        help="SpikeCount strength below and above the target")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--trial-ms", type=int, default=None,
                        help="Compress every recording into this many ms (default: real time)")
    length.add_argument("--crop-ms", type=int, default=None,
                        help="Keep only the first this many ms of every recording, in real time")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Reuse encoded recordings stored in this directory")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Seed for weight initialisation and shuffling (must not be 0)")
    parser.add_argument("--train-all", action="store_true",
                        help="Train on all official training subjects, without validation")
    parser.add_argument("--lambda-epochs", type=int, nargs="*", default=[],
                        help="Epochs at which to record lambda_V of one training example")
    parser.add_argument("--delay-layers", nargs="*", choices=LAYER_NAMES, default=[],
                        help="Layers whose incoming connection learns delays")
    parser.add_argument("--max-delay-ms", type=float, default=MAX_DELAY_MS)
    parser.add_argument("--init-delay-ms", type=float, default=INIT_DELAY_MS,
                        help="Initial delays are drawn from U(0, this)")
    parser.add_argument("--delay-lr", type=float, default=DELAY_LEARNING_RATE,
                        help="Adam learning rate for delays (in timesteps)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Initial (after warm-up: peak) Adam learning rate for weights")
    parser.add_argument("--output-lr-factor", type=float, default=1.0,
                        help="Scale the weight learning rate of the output layer by this")
    parser.add_argument("--channels", type=int, nargs=3, default=CHANNELS,
                        metavar=("CONV1", "CONV2", "CONV3"), help="Filters of the conv layers")
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="constant",
                        help="cosine decays every learning rate towards 0 at --epochs")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Ramp the learning rates up linearly over this many epochs")
    parser.add_argument("--downsample", choices=DOWNSAMPLING, default="pool",
                        help="stride: stride-2 conv3 and a Dense hidden layer, no pooling; "
                             "conv3-stride: stride-2 conv3, pooled hidden layer")
    parser.add_argument("--no-hidden", action="store_true",
                        help="Drop the 256-neuron hidden layer (conv3 -> output)")
    parser.add_argument("--tau-mem-ms", type=float, default=TAU_MEM_MS,
                        help="Membrane time constant of the spiking layers")
    parser.add_argument("--tau-syn-ms", type=float, default=TAU_SYN_MS,
                        help="Synaptic time constant")
    parser.add_argument("--synapse", choices=SYNAPSES, default="exponential",
                        help="Synapse model of the spiking layers")
    parser.add_argument("--subtract-reset", action="store_true",
                        help="Subtract the threshold after a spike instead of resetting v to 0")
    parser.add_argument("--spiking-weight-sd", type=float, default=SPIKING_WEIGHT_SD,
                        help="Initial weight sd of the spiking layers (scale with tau_mem)")
    parser.add_argument("--evd-tau-theta-ms", type=float, default=TAU_THETA_MS,
                        help="EvDownsampling adaptive-threshold time constant")
    parser.add_argument("--evd-tau-accumulator-ms", type=float, default=TAU_ACCUMULATOR_MS,
                        help="EvDownsampling accumulator time constant")
    parser.add_argument("--evd-beta", type=float, default=BETA,
                        help="EvDownsampling threshold adaptation scale")
    parser.add_argument("--aug-shift-px", type=float, default=0.0,
                        help="Shift each training recording by up to this many input pixels")
    parser.add_argument("--aug-zoom", type=float, nargs=2, default=(1.0, 1.0),
                        help="Range of the random zoom factor of the training recordings")
    parser.add_argument("--aug-rotate-deg", type=float, default=0.0,
                        help="Rotate each training recording by up to this many degrees")

    known, _ = parser.parse_known_args()
    if known.config is not None:
        try:
            apply_config(parser, load_config(known.config, known.run))
        except ValueError as error:
            parser.error(str(error))
    elif known.run is not None:
        parser.error("--run needs --config")
    args = parser.parse_args()
    args.reg_strength = tuple(args.reg_strength)
    if args.seed == 0:
        parser.error("--seed 0 makes GeNN pick a random seed")
    if args.delay_layers and not 0 < args.init_delay_ms <= args.max_delay_ms:
        parser.error("--init-delay-ms must be in (0, --max-delay-ms]")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("--warmup-epochs must be in [0, --epochs)")
    args.aug_zoom = tuple(args.aug_zoom)
    if (args.aug_shift_px < 0 or args.aug_rotate_deg < 0
            or not 0 < args.aug_zoom[0] <= args.aug_zoom[1]):
        parser.error("--aug-shift-px and --aug-rotate-deg must be >= 0, "
                     "--aug-zoom needs 0 < low <= high")
    args.channels = tuple(args.channels)
    if args.output_lr_factor <= 0 or min(args.channels) < 1:
        parser.error("--output-lr-factor must be > 0 and --channels >= 1")
    augment = args.aug_shift_px > 0 or args.aug_rotate_deg > 0 or args.aug_zoom != (1.0, 1.0)

    np.random.seed(args.seed)
    config_path = str(args.config.resolve()) if args.config else None
    if config_path:
        print(f"Config: {config_path}" + (f", run {args.run}" if args.run else ""))

    dataset = load_dataset(args.data_root.resolve(), train=True)
    train_idx, val_idx = split_train_validation(dataset)
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    evd_params = (args.evd_tau_theta_ms, args.evd_tau_accumulator_ms, args.evd_beta)
    train_x, train_y, train_len = encode_recordings(dataset, train_idx, args.encoding,
                                                    "Encoding training", args.trial_ms,
                                                    cache_dir, args.crop_ms, evd_params)
    val_x, val_y, val_len = encode_recordings(dataset, val_idx, args.encoding,
                                              "Encoding validation", args.trial_ms,
                                              cache_dir, args.crop_ms, evd_params)

    example_timesteps = int(max(train_len.max(), val_len.max()))
    if args.train_all:
        train_x = train_x + val_x
        train_y = np.concatenate((train_y, val_y))
        val_x, val_y = [], val_y[:0]
    growth = augmentation_growth(args.aug_zoom, args.aug_rotate_deg) if augment else 1
    input_buffer = BATCH_SIZE * calc_max_spikes(train_x + val_x) * growth
    peak_input = max_spikes_per_neuron(train_x + val_x)
    train_spikes = sum(len(s.spike_times) for s in train_x)
    print(f"Train: {len(train_x)}, validation: {len(val_x)}, encoding: {args.encoding}")
    print(f"Trial: {example_timesteps} ms, input buffer: {input_buffer} spikes, "
          f"busiest input neuron: {peak_input} spikes per gesture")
    if augment:
        print(f"Augmentation: shift +-{args.aug_shift_px} px, zoom {args.aug_zoom[0]}-"
              f"{args.aug_zoom[1]}, rotation +-{args.aug_rotate_deg} deg")
    augment_rng = np.random.default_rng(args.seed)

    if 2 * peak_input + 2 > MAX_SPIKES:
        raise ValueError(f"An input neuron fires {peak_input} times; "
                         f"MAX_SPIKES={MAX_SPIKES} must be at least {2 * peak_input + 2}")

    args.run_dir.mkdir(parents=True)
    os.chdir(args.run_dir)

    Path("run_config.json").write_text(json.dumps(dict(
        encoding=args.encoding, trial_ms=args.trial_ms, crop_ms=args.crop_ms,
        epochs=args.epochs,
        example_timesteps=example_timesteps,
        input_buffer=input_buffer, batch_size=BATCH_SIZE, max_spikes=MAX_SPIKES,
        grad_limit=args.grad_limit, reg_target=args.reg_target,
        reg_strength=args.reg_strength, seed=args.seed, train_all=args.train_all,
        lambda_epochs=args.lambda_epochs, delay_layers=args.delay_layers,
        max_delay_ms=args.max_delay_ms, init_delay_ms=args.init_delay_ms,
        delay_lr=args.delay_lr, lr=args.lr, output_lr_factor=args.output_lr_factor,
        channels=args.channels, lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs, downsample=args.downsample,
        hidden=not args.no_hidden, tau_mem_ms=args.tau_mem_ms, tau_syn_ms=args.tau_syn_ms,
        synapse=args.synapse, subtract_reset=args.subtract_reset,
        spiking_weight_sd=args.spiking_weight_sd, evd_params=evd_params,
        aug_shift_px=args.aug_shift_px, aug_zoom=args.aug_zoom,
        aug_rotate_deg=args.aug_rotate_deg,
        config=config_path, run=args.run,
        cache_dir=str(cache_dir) if cache_dir else None), indent=2))

    network = build_network(input_buffer, args.delay_layers, args.max_delay_ms,
                            args.init_delay_ms, args.downsample, not args.no_hidden,
                            args.tau_mem_ms, args.tau_syn_ms, args.spiking_weight_sd,
                            args.channels, args.synapse, args.subtract_reset)
    print("Layers: " + ", ".join(f"{p.name} {p.shape}" for p in network.populations))
    input_pop, output_pop = network.populations[0], network.populations[-1]
    hidden_layers = network.populations[1:-1]
    compiled = compile_for_training(network, example_timesteps, args.grad_limit,
                                    args.reg_strength, args.reg_target, args.seed,
                                    args.delay_lr, args.lr, args.output_lr_factor)

    metrics_log = CSVLog("metrics.csv", ["epoch", "train_correct", "train_total", "train_loss",
                                         "val_correct", "val_total", "seconds", "lr"])
    spikes_log = CSVLog("spikes.csv", ["epoch", "split", "layer", "mean_per_neuron",
                                       "max_per_neuron", "silent_fraction"])
    limit_log = CSVLog("adjoint_limits.csv", ["epoch", "layer", "limit", "zero_fraction"])
    if args.lambda_epochs:
        lambda_log = CSVLog("lambda_trace.csv", [
            "epoch", "layer", "limit", "forward_spikes", "zero_fraction", "zero_neurons",
            "clipped_jumps", "clipped_per_spike", "median_nonzero", "max_abs"])
    delay_log = None
    if args.delay_layers:
        delay_log = CSVLog("delays.csv", ["epoch", "layer", "count", "mean", "sd", "p50",
                                          "p95", "at_zero", "at_max"])
    checkpoint = Checkpoint(Numpy("checkpoints"))
    schedule = lr_schedule(args.lr_schedule, args.epochs, args.warmup_epochs)

    with compiled:
        compiled.save(("initial",), Numpy("checkpoints"))
        for epoch in range(args.epochs):
            start = perf_counter()
            epoch_x, kept = train_x, ""
            if augment:
                epoch_x = augment_spikes(train_x, augment_rng, args.aug_shift_px,
                                         args.aug_zoom, args.aug_rotate_deg)
                if BATCH_SIZE * calc_max_spikes(epoch_x) > input_buffer:
                    raise ValueError(f"Augmented input needs more than {input_buffer} spikes")
                kept = sum(len(s.spike_times) for s in epoch_x) / train_spikes
                kept = f", augmented input keeps {kept:.1%} of the spikes"
            limit_logger = AdjointLimitLog(limit_log, hidden_layers)
            callbacks = ["batch_progress_bar", checkpoint, schedule, limit_logger,
                         *spike_recorders(hidden_layers, "train", len(train_x))]
            if delay_log is not None:
                callbacks.append(DelayLog(delay_log, delay_connections(network)))
            trace = epoch in args.lambda_epochs
            if trace:
                limit_recorder = BackwardLimitRecorder(hidden_layers)
                callbacks += [limit_recorder, *lambda_recorders(hidden_layers)]
            if args.train_all:
                train_metrics, train_data = compiled.train(
                    {input_pop: epoch_x}, {output_pop: train_y},
                    num_epochs=1, start_epoch=epoch, shuffle=True, callbacks=callbacks)
                val_metrics, val_data = None, None
            else:
                train_metrics, val_metrics, train_data, val_data = compiled.train_validate(
                    {input_pop: epoch_x}, {output_pop: train_y},
                    validation_x={input_pop: val_x}, validation_y={output_pop: val_y},
                    num_epochs=1, start_epoch=epoch, shuffle=True, callbacks=callbacks,
                    validation_callbacks=spike_recorders(hidden_layers, "validation",
                                                         len(val_x)))

            train = train_metrics[output_pop]

            train_loss = float(np.mean(train_data["loss"]))
            if not np.isfinite(train_loss):
                raise FloatingPointError(f"Training loss is {train_loss} in epoch {epoch}")
            val_correct = val_total = ""
            if val_metrics is not None:
                val_correct = val_metrics[output_pop].correct
                val_total = val_metrics[output_pop].total
            metrics_log.write(epoch=epoch, train_correct=train.correct, train_total=train.total,
                              train_loss=train_loss, val_correct=val_correct,
                              val_total=val_total, seconds=round(perf_counter() - start, 1),
                              lr=compiled.optimiser_state[0].alpha)
            log_spike_counts(spikes_log, epoch, "train", train_data, hidden_layers)
            if trace:
                log_lambda_trace(lambda_log, epoch, train_data, limit_recorder.limits,
                                 hidden_layers)
            if val_data is not None:
                log_spike_counts(spikes_log, epoch, "validation", val_data, hidden_layers)
            validation = f", validation {val_correct}/{val_total}" if val_total else ""
            print(f"Epoch {epoch}: train {train.correct}/{train.total}{validation}, "
                  f"train loss {train_loss:.4f}{kept}")
            if limit_logger.blocked:
                raise RuntimeError(f"LambdavLimit of {', '.join(limit_logger.blocked)} is 0 after "
                                   f"epoch {epoch}: no loss gradient passes through it any more")


if __name__ == "__main__":
    main()
