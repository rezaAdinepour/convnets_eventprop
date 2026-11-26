import json
from pathlib import Path

PAPER = "Arfa et al."
LAYER_TITLES = {"conv1": "Conv1", "conv2": "Conv2", "conv3": "Conv3", "hidden": "Hidden",
                "output": "Output", "recurrent": "Recurrent"}


def load_config(run_dir):
    return json.loads((Path(run_dir) / "run_config.json").read_text())


def run_label(config):
    kind = "Final" if config.get("train_all") else "Val. split"
    return f"{kind} (seed {config['seed']})"


def initial_label(config):
    return f"Initial (seed {config['seed']})"


def layer_title(name):
    return LAYER_TITLES.get(name, name)
