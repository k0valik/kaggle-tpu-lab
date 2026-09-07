import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from kernel import serve_qwen38_gpu
from tools import sync_gpu_notebook


class GpuNotebookPreflightTest(unittest.TestCase):
    def test_preflight_creates_configured_scratch_directory(self):
        """Kaggle does not pre-create /kaggle/tmp; preflight must create scratch."""
        with tempfile.TemporaryDirectory() as parent:
            scratch = Path(parent) / "missing" / "scratch"
            fake_gpus = "0, Tesla T4, 15360\n1, Tesla T4, 15360\n"
            fake_usage = SimpleNamespace(free=40 * 1024**3)
            with (
                mock.patch.dict(os.environ, {"QWEN38_SCRATCH": str(scratch)}),
                mock.patch(
                    "subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout=fake_gpus,
                        stderr="",
                    ),
                ),
                mock.patch("shutil.disk_usage", return_value=fake_usage) as disk_usage,
            ):
                exec(sync_gpu_notebook.GPU_CHECK, {})

            self.assertTrue(scratch.is_dir())
            disk_usage.assert_called_once_with(scratch)

    def test_self_test_rejects_an_empty_generation(self):
        response = {"choices": [{"message": {"content": "", "reasoning_content": ""}}]}
        with mock.patch.object(serve_qwen38_gpu, "api_request", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "empty response"):
                serve_qwen38_gpu.self_test()


if __name__ == "__main__":
    unittest.main()
