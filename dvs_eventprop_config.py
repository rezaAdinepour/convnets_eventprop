"""TASK 09 first diagnostic configuration; no scientific training result."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProbeConfig:
    frames: int = 128
    dt_ms: float = 1.0
    seed: int = 42
    learning_rate: float = 0.001

    @property
    def tape_capacity(self):
        # Conservative previous-trial + current-trial bound, plus sentinel.
        return 2 * self.frames + 1

    def record(self):
        return {
            **asdict(self),
            "profile": "dvs-occ1ms-prefix128-diagnostic-v1",
            "input_hwp": [32, 32, 2],
            "hidden_hwf": [7, 7, 8],
            "hidden_flat": 392,
            "output_classes": 11,
            "kernel_hwio": [5, 5, 2, 8],
            "dense_io": [392, 11],
            "conv_stride": [4, 4],
            "conv_padding": "valid",
            "tau_mem_ms": 20.0,
            "tau_syn_ms": 5.0,
            "absolute_reset_voltage": 0.0,
            "threshold": 1.0,
            "readout": "avg_var_exp_weight",
            "loss": "sparse_categorical_crossentropy",
            "per_timestep_loss": False,
            "regulariser": None,
            "grad_limit": 100.0,
            "compiler_max_spikes": self.tape_capacity,
            "strict_buffer_checking": True,
        }
