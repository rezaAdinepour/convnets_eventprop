"""Minimal selected architecture A, using the inspected local mlGeNN APIs."""

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Conv2D, Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import SpikeInput, LeakyIntegrateFire, LeakyIntegrate
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential


def build_model(config, batch_size):
    network = SequentialNetwork()

    with network:
        input_layer = InputLayer(
            SpikeInput(max_spikes=batch_size * 2048 * config.frames),
            shape=(32, 32, 2),
            name="input",
            record_spikes=True,
        )

        hidden = Layer(
            Conv2D(
                Normal(mean=0.1, sd=0.5),
                filters=8,
                conv_size=5,
                flatten=True,
                conv_strides=4,
                conv_padding="valid",
                delay=0,
            ),
            LeakyIntegrateFire(
                v_thresh=1.0,
                v_reset=0.0,
                v=0.0,
                tau_mem=20.0,
                relative_reset=False,
            ),
            synapse=Exponential(tau=5.0),
            name="hidden",
            record_spikes=True,
        )

        output = Layer(
            Dense(Normal(mean=0.0, sd=0.1), delay=0),
            LeakyIntegrate(
                v=0.0,
                bias=0.0,
                tau_mem=20.0,
                readout="avg_var_exp_weight",
            ),
            shape=11,
            synapse=Exponential(tau=5.0),
            name="output",
        )

    if input_layer.population().shape != (32, 32, 2):
        raise RuntimeError("Unexpected input shape")
    if hidden.population().shape != (392,):
        raise RuntimeError("Unexpected flattened convolution shape")
    if output.population().shape != (11,):
        raise RuntimeError("Unexpected output shape")

    return network, input_layer, hidden, output


def compile_model(config, batch_size, *, learn, name):
    network, input_layer, hidden, output = build_model(config, batch_size)

    compiler = EventPropCompiler(
        example_timesteps=config.frames,
        losses="sparse_categorical_crossentropy",
        grad_limit=100.0,
        max_spikes=config.tape_capacity,
        strict_buffer_checking=True,
        per_timestep_loss=False,
        dt=config.dt_ms,
        softmax_temperature=1.0,
        batch_size=batch_size,
        rng_seed=config.seed,
        kernel_profiling=False,
    )

    # Empty mapping explicitly disables optimization for delivery checks.
    optimisers = (
        {
            hidden: {"weight": Adam(alpha=config.learning_rate)},
            output: {"weight": Adam(alpha=config.learning_rate)},
        }
        if learn else {}
    )

    compiled = compiler.compile(
        network,
        name=name,
        optimisers=optimisers,
        regularisers={},
    )
    return network, compiled, input_layer, hidden, output
