from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import AvgPoolConv2D, Conv2D, Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential

from data import INPUT_SHAPE, NUM_CLASSES, TRIAL_STEPS
from evdownsampling import BIN_MS

CONV_FILTERS = (16, 16, 8)
HIDDEN_UNITS = 256
KERNEL_SIZES = (5, 3, 3, 3)
CONV_STRIDES = (2, 1, 1, 1)
PADDING = ("valid", "same", "same", "valid")
POOL_SIZE = 2
POOL_SUM = True
WEIGHT_MEAN = 0.0
WEIGHT_SD = 1.0
V_INITIAL = 0.0
V_THRESHOLD = 1.0
V_RESET = 0.0
RELATIVE_RESET = False
OUTPUT_BIAS = 0.0
TAU_MEM_MS = 20.0
TAU_SYN_MS = 5.0
READOUT = "avg_var_exp_weight"
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
GRAD_LIMIT = 100.0
SOFTMAX_TEMPERATURE = 1.0
PER_TIMESTEP_LOSS = False
DT_MS = BIN_MS 
BATCH_SIZE = 32
SINGLE_EXAMPLE = 1
STRICT_BUFFER_CHECKING = True
BATCH_TAPE_SLOTS = 4096
GENN_SEED = 0
BACKEND = "cuda"
MODEL_NAME = "DVSGesture"
KERNEL_PROFILING = False
RESET_IN_SYN_BETWEEN_BATCHES = True
PREFIX_INPUT_CAPACITY = 1_000_000
PREFIX_TAPE_SLOTS = 500
ADJACENT_TRIALS = 2
TAPE_SPARE_SLOTS = 2

EXPECTED_CONV_SHAPES = ((15, 15, CONV_FILTERS[0]), (15, 15, CONV_FILTERS[1]), (7, 7, CONV_FILTERS[2]))
EXPECTED_POOL_SHAPES = ((7, 7, CONV_FILTERS[1]), (3, 3, CONV_FILTERS[2]))


def weights():
    return Normal(mean=WEIGHT_MEAN, sd=WEIGHT_SD)


def hidden_neuron():
    return LeakyIntegrateFire(v=V_INITIAL, v_thresh=V_THRESHOLD, v_reset=V_RESET,
                              tau_mem=TAU_MEM_MS, relative_reset=RELATIVE_RESET)


def build_model(input_capacity=None):
    network = SequentialNetwork()
    with network:
        input_neuron = SpikeInput(max_spikes=PREFIX_INPUT_CAPACITY if input_capacity is None else input_capacity)
        InputLayer(input_neuron, INPUT_SHAPE, name="input")
        for i, name in enumerate(("conv1", "conv2")):
            Layer(Conv2D(weights(), CONV_FILTERS[i], KERNEL_SIZES[i],
                         conv_strides=CONV_STRIDES[i], conv_padding=PADDING[i]),
                  hidden_neuron(), synapse=Exponential(TAU_SYN_MS), name=name)
        Layer(AvgPoolConv2D(weights(), CONV_FILTERS[2], pool_size=POOL_SIZE, conv_size=KERNEL_SIZES[2],
                            conv_strides=CONV_STRIDES[2], conv_padding=PADDING[2], sum=POOL_SUM),
              hidden_neuron(), synapse=Exponential(TAU_SYN_MS), name="conv3")
        Layer(AvgPoolConv2D(weights(), HIDDEN_UNITS, pool_size=POOL_SIZE, conv_size=KERNEL_SIZES[3],
                            conv_strides=CONV_STRIDES[3], conv_padding=PADDING[3], sum=POOL_SUM, flatten=True),
              hidden_neuron(), synapse=Exponential(TAU_SYN_MS), name="hidden")
        Layer(Dense(weights()), LeakyIntegrate(v=V_INITIAL, bias=OUTPUT_BIAS, tau_mem=TAU_MEM_MS, readout=READOUT),
              shape=NUM_CLASSES, synapse=Exponential(TAU_SYN_MS), name="output")
    return network


def compile_model(network, trial_steps=TRIAL_STEPS, *, whole=False,
                  batch_size=SINGLE_EXAMPLE, ring_slots=None, grad_limit=GRAD_LIMIT):
    """Single-example diagnostics retain their original tape allocation."""
    if ring_slots is None:
        ring_slots = tape_slots(trial_steps) if whole else PREFIX_TAPE_SLOTS
    compiler = EventPropCompiler(example_timesteps=trial_steps, losses=LOSS, backend=BACKEND,
                                 dt=DT_MS, batch_size=batch_size, rng_seed=GENN_SEED,
                                 kernel_profiling=KERNEL_PROFILING, grad_limit=grad_limit,
                                 per_timestep_loss=PER_TIMESTEP_LOSS, softmax_temperature=SOFTMAX_TEMPERATURE,
                                 max_spikes=ring_slots, strict_buffer_checking=STRICT_BUFFER_CHECKING)
    adam = Adam(alpha=LEARNING_RATE, beta1=ADAM_BETA1, beta2=ADAM_BETA2, epsilon=ADAM_EPSILON)
    return compiler.compile(network, name=MODEL_NAME, optimisers={"all_connections": {"weight": adam}})


def tape_slots(steps):
    return ADJACENT_TRIALS * steps + TAPE_SPARE_SLOTS


def inference_compiler(steps, batch_size=SINGLE_EXAMPLE):
    return InferenceCompiler(evaluate_timesteps=steps, dt=DT_MS, backend=BACKEND,
                             batch_size=batch_size, rng_seed=GENN_SEED, kernel_profiling=KERNEL_PROFILING,
                             reset_in_syn_between_batches=RESET_IN_SYN_BETWEEN_BATCHES)


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-in", type=Path, metavar="NEW_DIRECTORY",
                        help="Build and load the CUDA training model in a new directory")
    args = parser.parse_args()
    network = build_model()
    shapes = [pop.shape for pop in network.populations]
    if shapes != [INPUT_SHAPE, *EXPECTED_CONV_SHAPES, (HIDDEN_UNITS,), (NUM_CLASSES,)]:
        raise ValueError(f"Unexpected population shapes: {shapes}")
    pooled = [conn.connectivity.pool_output_shape for conn in network.connections
              if isinstance(conn.connectivity, AvgPoolConv2D)]
    if tuple(pooled) != EXPECTED_POOL_SHAPES:
        raise ValueError(f"Invalid network geometry: {shapes}, {pooled}")
    for pop in network.populations:
        print(f"{pop.name}: {pop.shape}")
    print(f"Fused pooling shapes: {pooled}")

    if args.compile_in is not None:
        args.compile_in.mkdir()
        os.chdir(args.compile_in)
        compiled = compile_model(network)
        with compiled:
            print(f"Loaded EventProp model: backend={compiled.genn_model.backend_name}, "
                  f"populations={len(compiled.neuron_populations)}, "
                  f"connections={len(compiled.connection_populations)}")
        print("CUDA build, load and unload: OK")
