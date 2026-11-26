"""Plain-language names of runs and layers for figures read by people outside the project."""
import json
from pathlib import Path

PAPER = "Arfa et al. (snnTorch, surrogate gradient)"
PAPER_SHORT = "Arfa et al."
TRAINING_SUBJECTS = 23
DEFAULT_VALIDATION_SUBJECTS = (21, 22, 23)
LAYER_TITLES = {"conv1": "Convolutional layer 1", "conv2": "Convolutional layer 2",
                "conv3": "Convolutional layer 3", "hidden": "Dense hidden layer",
                "output": "Output layer", "recurrent": "Recurrent hidden connection"}


def load_config(run_dir):
    run_dir = Path(run_dir)
    return json.loads((run_dir / "run_config.json").read_text())


def subjects(config):
    if config.get("train_all"):
        return f"all {TRAINING_SUBJECTS} training subjects"
    held_out = config.get("val_subjects") or DEFAULT_VALIDATION_SUBJECTS
    return (f"{TRAINING_SUBJECTS - len(held_out)} training subjects "
            f"(subjects {', '.join(map(str, held_out))} held out for validation)")


def model_kind(config):
    return "final model" if config.get("train_all") else "validation-split model"


def run_label(config, short=False):
    """'Ours, final model, seed 1' or the long form with epochs and training subjects."""
    if short:
        return f"Ours, {model_kind(config)}, seed {config['seed']}"
    return (f"Ours (EventProp), {model_kind(config)}: {config['epochs']} epochs on "
            f"{subjects(config)}, seed {config['seed']}")


def initial_label(config):
    return f"Ours, initial random weights, seed {config['seed']}"


def layer_title(name):
    return LAYER_TITLES.get(name, name)
