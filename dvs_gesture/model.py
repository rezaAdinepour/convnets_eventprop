"""ConvSNN for DVS Gesture, trained with mlGeNN's EventProp compiler.

The layer sizes follow the surrogate-gradient reference model in
deep-snn-deployment-on-spinnaker2-single-chip-using-nir (PTQ/Training/PTQ.py):

    input   34x34x2
    conv1   5x5, 16 filters, stride 2              -> 15x15x16
    conv2   3x3, 16 filters, same padding          -> 15x15x16
    conv3   2x2 sum-pool + 3x3, 8 filters, same    ->  7x7x8
    hidden  2x2 sum-pool + 3x3, 256 filters, flat  ->  256
    output  dense, 11 leaky integrators (no spikes)

Pooling is fused into the following convolution with AvgPoolConv2D(sum=True).
"""

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import AvgPoolConv2D, Conv2D, Dense
from ml_genn.initializers import Normal
from ml_genn.losses import SparseCategoricalCrossentropy
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential

from data import INPUT_SHAPE, NUM_CLASSES
from evdownsampling import BIN_MS

TAU_MEM_MS = 20.0
TAU_SYN_MS = 5.0
LEARNING_RATE = 0.001

# The 100-epoch batch-size-1 runs used Normal(0, 1) for every layer. At that
# scale some neurons fire >10,000 spikes per gesture, which does not fit in the
# EventProp spike buffer at batch size 32. sd = 0.375 for the four spiking
# layers is what the batch-32 run dvs_gesture_AcMYdMDZ used without overflowing.
SPIKING_WEIGHT_SD = 0.375
OUTPUT_WEIGHT_SD = 1.0

BATCH_SIZE = 32
# Spike-time slots per neuron (and per batch lane) for the backward pass. The
# buffer holds the current and the previous trial, so a neuron may fire at
# most about MAX_SPIKES / 2 times per gesture. 6242 is the largest value that
# has been run at batch size 32 on the 16 GB RTX 4080.
MAX_SPIKES = 6242
GRAD_LIMIT = 100.0  # mlGeNN default


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


def compile_for_training(network, example_timesteps, grad_limit=GRAD_LIMIT):
    compiler = EventPropCompiler(
        example_timesteps=example_timesteps,
        # record_key makes mlGeNN return the training loss of every batch.
        losses=SparseCategoricalCrossentropy(record_key="loss"),
        dt=BIN_MS, batch_size=BATCH_SIZE, max_spikes=MAX_SPIKES,
        strict_buffer_checking=True, grad_limit=grad_limit)
    adam = Adam(alpha=LEARNING_RATE)
    return compiler.compile(network, name="dvs_gesture_train",
                            optimisers={"all_connections": {"weight": adam}})


def compile_for_inference(network, example_timesteps):
    # EventProp clears synaptic input between trials; do the same here so a
    # reloaded checkpoint gives the same validation predictions as in training.
    compiler = InferenceCompiler(evaluate_timesteps=example_timesteps,
                                 dt=BIN_MS, batch_size=BATCH_SIZE,
                                 reset_in_syn_between_batches=True)
    return compiler.compile(network, name="dvs_gesture_inference")
