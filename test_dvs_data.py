"""CPU synthetic acceptance tests. No dataset arrays are opened or written."""

from dataclasses import replace
from pathlib import Path
import runpy
import unittest

import numpy as np

from dvs_data import (
    Config, EVENT_DTYPE, assert_disjoint, canonical_events, denoise,
    file_hash, pack_occupancy, preprocess, sample_from_path,
    validate_payload,
)


PINNED_DENOISER = Path(
    "/its/home/ra679/Research/Experiments/reference_env_1HOjb99A/"
    "venv/lib/python3.10/site-packages/tonic/functional/denoise.py"
)


def raw(rows):
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def decoded(payload):
    """Return neuron/time pairs from cumulative intervals."""
    result = []
    start = 0
    for neuron, stop in enumerate(payload.end_spikes):
        result.extend(
            (neuron, int(t)) for t in payload.spike_times[start:int(stop)]
        )
        start = int(stop)
    return result


class DataContractTests(unittest.TestCase):
    def test_timestamp_conversion_and_source_preservation(self):
        source = raw([[1, 2, 0, 0.028], [1, 2, 1, 1.2345]])
        original = source.copy()
        events = canonical_events(source, chunk_size=1)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(events.dtype, EVENT_DTYPE)
        np.testing.assert_array_equal(events["t"], [28, 1234])
        np.testing.assert_array_equal(events["p"], [False, True])

    def test_ordering_equal_times_and_chunk_boundary(self):
        canonical_events(raw([[0, 0, 0, 2], [1, 0, 1, 2]]), chunk_size=1)
        with self.assertRaises(ValueError):
            canonical_events(
                raw([[0, 0, 0, 2], [1, 0, 1, 1]]), chunk_size=1
            )

    def test_invalid_schema_coordinates_polarity_and_time(self):
        bad_rows = [
            [0.5, 0, 0, 0],
            [-1, 0, 0, 0],
            [128, 0, 0, 0],
            [0, 128, 0, 0],
            [0, 0, 2, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [0, 0, 0, np.nan],
            [0, 0, 0, np.inf],
            [0, 0, 0, 1e30],
        ]
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                canonical_events(raw([row]))
        with self.assertRaises(ValueError):
            canonical_events(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            canonical_events(np.zeros((2, 4), dtype=object))

    def test_half_open_bins_duplicate_collapse_and_last_step(self):
        payload, audit = pack_occupancy(
            np.array([0, 999, 1000, 127999, 128000], dtype=np.int64),
            np.ones(5, dtype=np.int64),
            np.full(5, 2, dtype=np.int64),
            np.ones(5, dtype=np.int64),
            width=32, height=32, frames=128,
        )
        self.assertEqual(decoded(payload), [(131, 0), (131, 1), (131, 127)])
        self.assertEqual(audit["included_events"], 4)
        self.assertEqual(audit["prefix_excluded_events"], 1)
        self.assertEqual(audit["collapsed_excess"], 1)
        self.assertEqual(audit["max_cell_count"], 2)
        self.assertEqual(payload.end_spikes.dtype, np.uint32)
        self.assertEqual(payload.spike_times.dtype, np.float32)

    def test_rectangular_layout_and_separate_polarities(self):
        payload, _ = pack_occupancy(
            np.array([0, 0], dtype=np.int64),
            np.array([4, 4]), np.array([1, 1]), np.array([0, 1]),
            width=5, height=3, frames=128,
        )
        self.assertEqual(decoded(payload), [(18, 0), (19, 0)])
        self.assertEqual(payload.end_spikes.shape, (30,))

    def test_empty_recording_and_empty_filtered_recording(self):
        payload, metadata = preprocess(canonical_events(raw([])), Config(128))
        self.assertEqual(len(payload.spike_times), 0)
        self.assertTrue((payload.end_spikes == 0).all())
        self.assertIsNone(metadata["origin_us"])
        self.assertEqual(metadata["origin_kind"], "empty_no_observed_origin")
        self.assertTrue(metadata["raw_empty"])

        # A one-pixel recording has no axial neighbors in the pinned denoiser.
        events = canonical_events(raw([[0, 0, 1, 20.0]]))
        payload, metadata = preprocess(events, Config(128))
        self.assertEqual(len(payload.spike_times), 0)
        self.assertEqual(metadata["origin_us"], 20000)
        self.assertEqual(
            metadata["origin_kind"], "first_loaded_empty_filtered"
        )
        self.assertEqual(metadata["denoise_removed_events"], 1)
        self.assertFalse(metadata["raw_empty"])
        self.assertTrue(metadata["filtered_empty"])

    def test_empty_window_and_payload_validation(self):
        payload, audit = pack_occupancy(
            np.array([128000], dtype=np.int64),
            np.array([0]), np.array([0]), np.array([0]),
            width=32, height=32, frames=128,
        )
        self.assertTrue(audit["empty_prefix"])
        self.assertEqual(audit["prefix_excluded_events"], 1)
        validate_payload(payload, neurons=2048, frames=128)
        bad = payload._replace(spike_times=np.array([128], dtype=np.float32))
        with self.assertRaises(ValueError):
            validate_payload(bad, neurons=2048, frames=128)

    def test_class_ids_and_fixed_subject_membership(self):
        train = sample_from_path("ibmGestureTrain/user01_fluorescent/0.npy")
        validation = sample_from_path("ibmGestureTrain/user21_led/10.npy")
        test = sample_from_path("ibmGestureTest/user24_natural/5.npy")
        self.assertEqual((train.label, validation.label, test.label), (0, 10, 5))
        self.assertEqual(
            (train.split, validation.split, test.split),
            ("train", "validation", "test"),
        )
        for path in (
            "ibmGestureTrain/user01_led/11.npy",
            "ibmGestureTrain/user24_led/0.npy",
            "ibmGestureTest/user23_led/0.npy",
            "ibmGestureTrain/user01_led/00.npy",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                sample_from_path(path)

    def test_no_train_validation_test_overlap(self):
        train = sample_from_path("ibmGestureTrain/user01_led/0.npy")
        validation = sample_from_path("ibmGestureTrain/user21_led/0.npy")
        test = sample_from_path("ibmGestureTest/user24_led/0.npy")
        assert_disjoint({
            "train": [train], "validation": [validation], "test": [test]
        })
        with self.assertRaises(ValueError):
            assert_disjoint({
                "train": [train],
                "validation": [replace(validation, subject=train.subject)],
                "test": [test],
            })
        with self.assertRaises(ValueError):
            assert_disjoint({"train": [train, train], "test": [test]})
        with self.assertRaises(ValueError):
            assert_disjoint({"train": [train], "test": [train]})

    def test_pinned_denoiser_equivalence(self):
        self.assertTrue(PINNED_DENOISER.is_file())
        reference = runpy.run_path(str(PINNED_DENOISER))["denoise_numpy"]
        print("\nPINNED_DENOISER_SHA256", file_hash(PINNED_DENOISER))
        fixtures = [
            # Initial timestamp memory, equal times, edge pixels, polarity.
            raw([[0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]]),
            # Rejected events still update timestamp memory.
            raw([[0, 0, 0, 20], [1, 0, 1, 20.001]]),
            # Strict > comparison at filter-time boundary.
            raw([[0, 0, 0, 20], [1, 0, 1, 30]]),
            # Same coordinates do not substitute for an axial neighbor.
            raw([[0, 0, 0, 20], [0, 0, 1, 20.001]]),
        ]
        generator = np.random.default_rng(42)
        random_fixture = np.column_stack((
            generator.integers(0, 128, 2000),
            generator.integers(0, 128, 2000),
            generator.integers(0, 2, 2000),
            np.sort(generator.uniform(0, 100, 2000)),
        ))
        fixtures.append(random_fixture)
        for fixture in fixtures:
            events = canonical_events(fixture)
            np.testing.assert_array_equal(
                denoise(events), reference(events, filter_time=10000)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)