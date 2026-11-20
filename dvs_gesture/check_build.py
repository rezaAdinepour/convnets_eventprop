"""Find which combination fails to build when AvgPoolDense2D is used (no data needed)."""
import argparse
import os
import traceback
from pathlib import Path

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.connectivity import AvgPoolConv2D, AvgPoolDense2D, Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.synapses import Exponential

from data import NUM_CLASSES
from model import build_network, compile_for_inference, compile_for_training

TIMESTEPS = 100
INPUT_BUFFER = 1000


def small_network(with_conv):
    network = SequentialNetwork()
    with network:
        InputLayer(SpikeInput(max_spikes=INPUT_BUFFER), (15, 15, 16) if with_conv else (7, 7, 8))
        if with_conv:
            Layer(AvgPoolConv2D(Normal(), 8, 2, 3, conv_padding="same", sum=True),
                  LeakyIntegrateFire(), synapse=Exponential())
        Layer(AvgPoolDense2D(Normal(), 2, sum=True), LeakyIntegrateFire(), 256,
              synapse=Exponential())
        Layer(Dense(Normal()), LeakyIntegrate(readout="avg_var_exp_weight"), NUM_CLASSES,
              synapse=Exponential())
    return network


CASES = [
    ("full_inference", lambda: build_network(INPUT_BUFFER, dense_hidden=True),
     compile_for_inference),
    ("pool_dense_training", lambda: small_network(False), compile_for_training),
    ("conv_pool_dense_training", lambda: small_network(True), compile_for_training),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="New directory for GeNN's generated code")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True)
    os.chdir(args.out_dir)

    for name, make_network, compile_model in CASES:
        network = make_network()
        layers = " -> ".join(type(c.connectivity).__name__ for c in network.connections)
        try:
            with compile_model(network, TIMESTEPS, name=name):
                pass
            print(f"{name}: OK ({layers})")
        except Exception as error:
            print(f"{name}: FAILED ({layers}): {type(error).__name__}: {error}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
