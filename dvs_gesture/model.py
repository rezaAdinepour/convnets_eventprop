from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import AvgPoolConv2D, Conv2D, Dense
from ml_genn.initializers import Normal, Uniform
from ml_genn.losses import SparseCategoricalCrossentropy
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.regularisers import SpikeCount
from ml_genn.synapses import Exponential

from data import BIN_MS, INPUT_SHAPE, NUM_CLASSES

TAU_MEM_MS = 20.0
TAU_SYN_MS = 5.0
LEARNING_RATE = 0.001

SPIKING_WEIGHT_SD = 0.375
OUTPUT_WEIGHT_SD = 1.0

BATCH_SIZE = 32
SEED = 1

MAX_SPIKES = 6242
GRAD_LIMIT = 100.0

REG_TARGET = 100.0
REG_STRENGTH = (1e-10, 1e-10)

# Learnable delays (layer names whose incoming connection learns them).
# With kernel connectivity (Toeplitz or procedural, i.e. every layer but the
# Dense output) GeNN 5.5.0 fails in _create_backend with
# "vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)".
LAYER_NAMES = ("output",)
MAX_DELAY_MS = 150.0
INIT_DELAY_MS = 50.0
DELAY_LEARNING_RATE = 0.1

DOWNSAMPLING = ("pool", "stride", "conv3-stride")


def build_network(max_input_spikes_per_batch, delay_layers=(),
                  max_delay_ms=MAX_DELAY_MS, init_delay_ms=INIT_DELAY_MS,
                  downsample="pool", hidden=True, tau_mem_ms=TAU_MEM_MS, tau_syn_ms=TAU_SYN_MS):
    """Create the network. ``max_input_spikes_per_batch`` sizes the input buffer.

    ``tau_mem_ms`` applies to the spiking layers (the output readout keeps
    ``TAU_MEM_MS``), ``tau_syn_ms`` to every synapse.

    Connections into ``delay_layers`` get delays drawn from U(0, init_delay_ms)
    and a delay buffer of ``max_delay_ms``.

    ``downsample="stride"`` replaces the two 2x2 sum-pools: conv3 becomes a
    stride-2 valid 3x3 convolution (15x15 -> 7x7x8) and the hidden layer a Dense
    392 -> 256. ``"conv3-stride"`` strides conv3 only and keeps the pooled hidden
    layer. ``hidden=False`` drops the hidden layer (conv3 -> output).
    """
    def lif():
        return LeakyIntegrateFire(v_thresh=1.0, tau_mem=tau_mem_ms)

    def synapse():
        return Exponential(tau_syn_ms)

    def delay(name):
        # A constant delay would make mlGeNN skip the delayed adjoint lookup,
        # so the initial delays must be random (init_delay_ms > 0).
        if name not in delay_layers:
            return {}
        return dict(delay=Uniform(0.0, init_delay_ms / BIN_MS))

    def layer_delay(name):
        if name not in delay_layers:
            return {}
        return dict(max_delay_steps=round(max_delay_ms / BIN_MS))

    spiking_weights = Normal(mean=0.0, sd=SPIKING_WEIGHT_SD)
    network = SequentialNetwork()
    with network:
        InputLayer(SpikeInput(max_spikes=max_input_spikes_per_batch), INPUT_SHAPE, name="input")
        Layer(Conv2D(spiking_weights, 16, 5, conv_strides=2, **delay("conv1")),
              lif(), synapse=synapse(), record_spikes=True, name="conv1",
              **layer_delay("conv1"))
        Layer(Conv2D(spiking_weights, 16, 3, conv_padding="same", **delay("conv2")),
              lif(), synapse=synapse(), record_spikes=True, name="conv2",
              **layer_delay("conv2"))
        if downsample == "pool":
            conv3 = AvgPoolConv2D(spiking_weights, 8, 2, 3, conv_padding="same", sum=True,
                                  **delay("conv3"))
        else:
            conv3 = Conv2D(spiking_weights, 8, 3, conv_strides=2, **delay("conv3"))
        Layer(conv3, lif(), synapse=synapse(), record_spikes=True, name="conv3",
              **layer_delay("conv3"))
        if hidden:
            if downsample != "stride":
                connectivity = AvgPoolConv2D(spiking_weights, 256, 2, 3, sum=True,
                                             flatten=True, **delay("hidden"))
            else:
                connectivity = Dense(spiking_weights, **delay("hidden"))
            Layer(connectivity, lif(), 256, synapse=synapse(), record_spikes=True,
                  name="hidden", **layer_delay("hidden"))
        Layer(Dense(Normal(mean=0.0, sd=OUTPUT_WEIGHT_SD), **delay("output")),
              LeakyIntegrate(tau_mem=TAU_MEM_MS, readout="avg_var_exp_weight"),
              NUM_CLASSES, synapse=synapse(), name="output", **layer_delay("output"))
    return network


def delay_connections(network):
    """Connections built with a delay buffer, i.e. those that learn delays."""
    return [c for c in network.connections if c.max_delay_steps is not None]


def compile_for_training(network, example_timesteps, grad_limit=GRAD_LIMIT,
                         reg_strength=REG_STRENGTH, reg_target=REG_TARGET, seed=SEED,
                         delay_learning_rate=DELAY_LEARNING_RATE,
                         learning_rate=LEARNING_RATE):

    regulariser = SpikeCount(strength=reg_strength, target=reg_target)
    compiler = EventPropCompiler(
        example_timesteps=example_timesteps,

        losses=SparseCategoricalCrossentropy(record_key="loss"),
        dt=BIN_MS, batch_size=BATCH_SIZE, max_spikes=MAX_SPIKES,
        strict_buffer_checking=True, grad_limit=grad_limit, rng_seed=seed)
    adam = Adam(alpha=learning_rate)
    optimisers = {"all_connections": {"weight": adam}}
    for connection in delay_connections(network):
        optimisers[connection] = {"delay": Adam(alpha=delay_learning_rate)}
    return compiler.compile(network, name="dvs_gesture_train", optimisers=optimisers,
                            regularisers={"all_hidden_populations": regulariser})


def compile_for_inference(network, example_timesteps, seed=SEED):
    compiler = InferenceCompiler(evaluate_timesteps=example_timesteps,
                                 dt=BIN_MS, batch_size=BATCH_SIZE,
                                 reset_in_syn_between_batches=True, rng_seed=seed)
    return compiler.compile(network, name="dvs_gesture_inference")
