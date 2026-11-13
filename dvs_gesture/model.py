"""Construct the DVS Gesture network using public mlGeNN 3 layers."""

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import AvgPoolConv2D, Conv2D, Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.synapses import Exponential

from data import INPUT_SHAPE, TRIAL_STEPS


def build_model():
    network = SequentialNetwork()
    with network:
        InputLayer(SpikeInput(), INPUT_SHAPE, name="input")
        Layer(Conv2D(Normal(), 16, 5, conv_strides=2, conv_padding="valid"),
              LeakyIntegrateFire(), synapse=Exponential(), name="conv1")
        Layer(Conv2D(Normal(), 16, 3, conv_padding="same"),
              LeakyIntegrateFire(), synapse=Exponential(), name="conv2")
        Layer(AvgPoolConv2D(Normal(), 8, pool_size=2, conv_size=3,
                            conv_padding="same", sum=True),
              LeakyIntegrateFire(), synapse=Exponential(), name="conv3")
        # One valid kernel spans the entire pooled 3 x 3 field per output unit.
        Layer(AvgPoolConv2D(Normal(), 256, pool_size=2, conv_size=3,
                            conv_padding="valid", sum=True, flatten=True),
              LeakyIntegrateFire(), synapse=Exponential(), name="hidden")
        Layer(Dense(Normal()), LeakyIntegrate(readout="avg_var_exp_weight"),
              shape=11, synapse=Exponential(), name="output")
    return network


def compile_model(network):
    """Prepare the training model with the stock EventProp and Adam defaults."""
    compiler = EventPropCompiler(example_timesteps=TRIAL_STEPS,
                                 losses="sparse_categorical_crossentropy",
                                 backend="cuda")
    return compiler.compile(network, name="DVSGesture")


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
    expected = [INPUT_SHAPE, (15, 15, 16), (15, 15, 16), (7, 7, 8), (256,), (11,)]
    if shapes != expected:
        raise ValueError(f"Unexpected population shapes: {shapes}")
    pooled = [conn.connectivity.pool_output_shape for conn in network.connections
              if isinstance(conn.connectivity, AvgPoolConv2D)]
    if pooled != [(7, 7, 16), (3, 3, 8)]:
        raise ValueError(f"Unexpected pooling shapes: {pooled}")
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
