import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class ModelPowerReferenceTests(unittest.TestCase):
    def _fixtures(self):
        index = pd.Index(["chr1_10", "chr1_20", "chr2_30"], dtype=object)
        mean = pd.Series([0.10, 0.20, 0.30], index=index)
        std = pd.DataFrame(
            {
                "5_mean": [0.010, 0.020, 0.030],
                "5_CI_l": [0.009, 0.019, 0.029],
                "5_CI_u": [0.011, 0.021, 0.031],
                "10_mean": [0.015, 0.025, 0.035],
                "10_CI_l": [0.014, 0.024, 0.034],
                "10_CI_u": [0.016, 0.026, 0.036],
            },
            index=index,
        )
        return mean, std

    def test_convert_and_load_selected_depths_preserves_existing_api_shape(self):
        from analysis.model_power_reference import (
            convert_pickles_to_array_reference,
            load_model_power_reference,
        )

        mean, std = self._fixtures()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mean_path = root / "mean.pkl"
            std_path = root / "std.pkl"
            out_dir = root / "arrays"
            mean.to_pickle(mean_path)
            std.to_pickle(std_path)

            manifest = convert_pickles_to_array_reference(
                mean_path,
                std_path,
                out_dir,
                dtype="float32",
            )

            self.assertEqual(manifest["row_count"], 3)
            self.assertEqual(manifest["depths"], [5, 10])
            self.assertEqual(manifest["stats"], ["mean", "CI_l", "CI_u"])
            self.assertTrue((out_dir / "cpg_index.npy").exists())
            self.assertTrue((out_dir / "cpg_mean.float32.npy").exists())
            self.assertTrue(
                (out_dir / "std_by_depth" / "depth_10_mean.float32.npy").exists()
            )
            self.assertTrue(
                (out_dir / "std_by_depth" / "depth_10_CI.float32.npy").exists()
            )
            ci_array = np.load(
                out_dir / "std_by_depth" / "depth_10_CI.float32.npy"
            )
            self.assertEqual(ci_array.shape, (3, 2))

            loaded = load_model_power_reference(
                out_dir,
                depths=[10],
                sd_stats=["mean", "CI_u"],
                mmap_mode=None,
            )

            self.assertEqual(list(loaded.cpg_mean.index), list(mean.index))
            self.assertEqual(list(loaded.cpg_std_summary.index), list(std.index))
            self.assertEqual(
                list(loaded.cpg_std_summary.columns),
                ["10_mean", "10_CI_u"],
            )
            self.assertEqual(loaded.cpg_mean.dtype, np.float32)
            self.assertEqual(loaded.cpg_std_summary["10_mean"].dtype, np.float32)
            np.testing.assert_allclose(
                loaded.cpg_mean.to_numpy(),
                mean.to_numpy(dtype=np.float32),
            )
            np.testing.assert_allclose(
                loaded.cpg_std_summary["10_CI_u"].to_numpy(),
                std["10_CI_u"].to_numpy(dtype=np.float32),
            )

            manifest_data = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest_data["depths"], [5, 10])
            self.assertEqual(manifest_data["format_version"], 1)
            self.assertEqual(
                manifest_data["std_files"]["10"]["mean"],
                "std_by_depth/depth_10_mean.float32.npy",
            )
            self.assertEqual(
                manifest_data["std_files"]["10"]["CI"],
                "std_by_depth/depth_10_CI.float32.npy",
            )

    def test_mean_only_load_does_not_load_ci_arrays(self):
        from analysis.model_power_reference import (
            convert_pickles_to_array_reference,
            load_model_power_reference,
        )

        mean, std = self._fixtures()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mean_path = root / "mean.pkl"
            std_path = root / "std.pkl"
            out_dir = root / "arrays"
            mean.to_pickle(mean_path)
            std.to_pickle(std_path)
            convert_pickles_to_array_reference(mean_path, std_path, out_dir)

            ci_path = out_dir / "std_by_depth" / "depth_10_CI.float32.npy"
            ci_path.unlink()

            loaded = load_model_power_reference(
                out_dir,
                depths=[10],
                sd_stats=["mean"],
                mmap_mode=None,
            )

            self.assertEqual(list(loaded.cpg_std_summary.columns), ["10_mean"])
            np.testing.assert_allclose(
                loaded.cpg_std_summary["10_mean"].to_numpy(),
                std["10_mean"].to_numpy(dtype=np.float32),
            )

    def test_include_index_false_does_not_load_cpg_index(self):
        from analysis.model_power_reference import (
            convert_pickles_to_array_reference,
            load_model_power_reference,
        )

        mean, std = self._fixtures()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mean_path = root / "mean.pkl"
            std_path = root / "std.pkl"
            out_dir = root / "arrays"
            mean.to_pickle(mean_path)
            std.to_pickle(std_path)
            convert_pickles_to_array_reference(mean_path, std_path, out_dir)

            (out_dir / "cpg_index.npy").unlink()

            loaded = load_model_power_reference(
                out_dir,
                depths=[5],
                sd_stats=["mean"],
                include_index=False,
                mmap_mode=None,
            )

            self.assertIsInstance(loaded.cpg_mean.index, pd.RangeIndex)
            self.assertEqual(list(loaded.cpg_std_summary.index), [0, 1, 2])
            np.testing.assert_allclose(
                loaded.cpg_mean.to_numpy(),
                mean.to_numpy(dtype=np.float32),
            )

    def test_convert_rejects_mismatched_indices(self):
        from analysis.model_power_reference import convert_pickles_to_array_reference

        mean, std = self._fixtures()
        std = std.copy()
        std.index = ["chr1_10", "chr2_30", "chr1_20"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mean_path = root / "mean.pkl"
            std_path = root / "std.pkl"
            mean.to_pickle(mean_path)
            std.to_pickle(std_path)

            with self.assertRaisesRegex(ValueError, "identical row order"):
                convert_pickles_to_array_reference(
                    mean_path,
                    std_path,
                    root / "arrays",
                )

    def test_loader_rejects_unknown_depth(self):
        from analysis.model_power_reference import (
            convert_pickles_to_array_reference,
            load_model_power_reference,
        )

        mean, std = self._fixtures()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mean_path = root / "mean.pkl"
            std_path = root / "std.pkl"
            out_dir = root / "arrays"
            mean.to_pickle(mean_path)
            std.to_pickle(std_path)
            convert_pickles_to_array_reference(mean_path, std_path, out_dir)

            with self.assertRaisesRegex(ValueError, "Depths not available"):
                load_model_power_reference(out_dir, depths=[20])


if __name__ == "__main__":
    unittest.main()
