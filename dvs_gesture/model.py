from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import AvgPoolConv2D, Conv2D, Dense
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

MAX_SPIKES = 6242
GRAD_LIMIT = 100.0

# Starting point from Jamie: target 100 spikes per neuron per gesture, strength
# from examples/event_prop/shd.py. The strength needs to be explored.
REG_TARGET = 100.0
REG_STRENGTH = (1e-10, 1e-10)


def build_network(max_input_spikes_per_batch):
    """Create the network. ``max_input_spikes_per_batch`` sizes the input buffer."""
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
        Layer(AvgPoolConv2D(spiking_weights, 256, 2, 3, sum=True, flatten=True),
              lif(), synapse=synapse(), record_spikes=True, name="hidden")
        Layer(Dense(Normal(mean=0.0, sd=OUTPUT_WEIGHT_SD)),
              LeakyIntegrate(tau_mem=TAU_MEM_MS, readout="avg_var_exp_weight"),
              NUM_CLASSES, synapse=synapse(), name="output")
    return network


def compile_for_training(network, example_timesteps, grad_limit=GRAD_LIMIT,
                         reg_strength=REG_STRENGTH, reg_target=REG_TARGET):
    # SpikeCount pulls each hidden neuron's spike count per gesture towards
    # reg_target; reg_strength is (below target, above target).
    regulariser = SpikeCount(strength=reg_strength, target=reg_target)
    compiler = EventPropCompiler(
        example_timesteps=example_timesteps,
        # record_key makes mlGeNN return the training loss of every batch.
        losses=SparseCategoricalCrossentropy(record_key="loss"),
        dt=BIN_MS, batch_size=BATCH_SIZE, max_spikes=MAX_SPIKES,
        strict_buffer_checking=True, grad_limit=grad_limit)
    adam = Adam(alpha=LEARNING_RATE)
    return compiler.compile(network, name="dvs_gesture_train",
                            optimisers={"all_connections": {"weight": adam}},
                            regularisers={"all_hidden_populations": regulariser})


def compile_for_inference(network, example_timesteps):
    compiler = InferenceCompiler(evaluate_timesteps=example_timesteps,
                                 dt=BIN_MS, batch_size=BATCH_SIZE,
                                 reset_in_syn_between_batches=True)
    return compiler.compile(network, name="dvs_gesture_inference")
