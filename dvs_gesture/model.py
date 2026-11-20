from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import AvgPoolConv2D, AvgPoolDense2D, Conv2D, Dense
from ml_genn.initializers import Normal
from ml_genn.losses import SparseCategoricalCrossentropy
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.regularisers import SpikeCount
from ml_genn.synapses import Exponential

from data import INPUT_SHAPE, NUM_CLASSES
from evdownsampling import BIN_MS

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


def build_network(max_input_spikes_per_batch, dense_hidden=False):
    """Create the network. ``max_input_spikes_per_batch`` sizes the input buffer.

    dense_hidden=True uses AvgPoolDense2D for the 256-unit layer, as in the
    reference model, instead of the equivalent full-field AvgPoolConv2D.
    """
    def lif():
        return LeakyIntegrateFire(v_thresh=1.0, tau_mem=TAU_MEM_MS)

    def synapse():
        return Exponential(TAU_SYN_MS)

    spiking_weights = Normal(mean=0.0, sd=SPIKING_WEIGHT_SD)
    network = SequentialNetwork()
    with network:
        InputLayer(SpikeInput(max_spikes=max_input_spikes_per_batch), INPUT_SHAPE, name="input")
        Layer(Conv2D(spiking_weights, 16, 5, conv_strides=2),
              lif(), synapse=synapse(), record_spikes=True, name="conv1")
        Layer(Conv2D(spiking_weights, 16, 3, conv_padding="same"),
              lif(), synapse=synapse(), record_spikes=True, name="conv2")
        Layer(AvgPoolConv2D(spiking_weights, 8, 2, 3, conv_padding="same", sum=True),
              lif(), synapse=synapse(), record_spikes=True, name="conv3")
        if dense_hidden:
            hidden, shape = AvgPoolDense2D(spiking_weights, 2, sum=True), 256
        else:
            hidden, shape = AvgPoolConv2D(spiking_weights, 256, 2, 3, sum=True, flatten=True), None
        Layer(hidden, lif(), shape, synapse=synapse(), record_spikes=True, name="hidden")
        Layer(Dense(Normal(mean=0.0, sd=OUTPUT_WEIGHT_SD)),
              LeakyIntegrate(tau_mem=TAU_MEM_MS, readout="avg_var_exp_weight"),
              NUM_CLASSES, synapse=synapse(), name="output")
    return network


def compile_for_training(network, example_timesteps, grad_limit=GRAD_LIMIT,
                         reg_strength=REG_STRENGTH, reg_target=REG_TARGET, seed=SEED):

    regulariser = SpikeCount(strength=reg_strength, target=reg_target)
    compiler = EventPropCompiler(
        example_timesteps=example_timesteps,

        losses=SparseCategoricalCrossentropy(record_key="loss"),
        dt=BIN_MS, batch_size=BATCH_SIZE, max_spikes=MAX_SPIKES,
        strict_buffer_checking=True, grad_limit=grad_limit, rng_seed=seed)
    adam = Adam(alpha=LEARNING_RATE)
    return compiler.compile(network, name="dvs_gesture_train",
                            optimisers={"all_connections": {"weight": adam}},
                            regularisers={"all_hidden_populations": regulariser})


def compile_for_inference(network, example_timesteps, seed=SEED):
    compiler = InferenceCompiler(evaluate_timesteps=example_timesteps,
                                 dt=BIN_MS, batch_size=BATCH_SIZE,
                                 reset_in_syn_between_batches=True, rng_seed=seed)
    return compiler.compile(network, name="dvs_gesture_inference")
