"""Build, load and unload the AvgPoolDense2D variant for training and inference (no data)."""
import argparse
import os
from pathlib import Path

from model import build_network, compile_for_inference, compile_for_training

TIMESTEPS = 100
INPUT_BUFFER = 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="New directory for GeNN's generated code")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True)
    os.chdir(args.out_dir)

    for stage, compile_model in (("training", compile_for_training),
                                 ("inference", compile_for_inference)):
        network = build_network(INPUT_BUFFER, dense_hidden=True)
        for pop in network.populations:
            print(f"  {pop.name}: {pop.shape}")
        for conn in network.connections:
            print(f"  {conn.source().name} -> {conn.target().name}: "
                  f"{type(conn.connectivity).__name__}")
        compiled = compile_model(network, TIMESTEPS)
        with compiled:
            print(f"{stage}: built and loaded on {compiled.genn_model.backend_name}")


if __name__ == "__main__":
    main()
