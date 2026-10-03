import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from qwen_decider.settings import Settings


class MemoryTests(unittest.TestCase):
    def test_small_fixed_defaults_and_rejected_overrides(self):
        settings = Settings(model_path=Path("model"))
        self.assertEqual(settings.max_num_seqs, 1)
        self.assertEqual(settings.kv_cache_memory_bytes, 128 * 1024**2)
        self.assertEqual(settings.gpu_memory_limit_bytes, 8_000_000_000)
        self.assertFalse(settings.enforce_eager)
        for changes in [
            {"max_num_seqs": 3},
            {"gpu_memory_limit_bytes": 8_000_000_001},
            {"kv_cache_memory_bytes": 512 * 1024**2},
        ]:
            with self.assertRaises(ValidationError):
                Settings(model_path=Path("model"), **changes)

    def test_nvml_guard_counts_only_own_process_and_rejects_excess(self):
        from qwen_decider.worker import MemoryBoundWorker

        worker = object.__new__(MemoryBoundWorker)
        worker.device = 0
        worker.memory_limit = 8_000_000_000
        worker.memory_peak = 0
        worker.handles = [0]
        rows = [
            SimpleNamespace(pid=os.getpid(), usedGpuMemory=5_000_000_000),
            SimpleNamespace(pid=-1, usedGpuMemory=20_000_000_000),
        ]
        worker.nvml = SimpleNamespace(nvmlDeviceGetComputeRunningProcesses=lambda _: rows)
        self.assertEqual(worker.check_memory()["used_bytes"], 5_000_000_000)
        rows[0].usedGpuMemory = 8_000_000_001
        with self.assertRaisesRegex(RuntimeError, "memory exceeded"):
            worker.check_memory()
        self.assertEqual(worker.memory_peak, 8_000_000_001)

    def test_execution_checks_memory_before_and_after_model(self):
        from qwen_decider.worker import MemoryBoundWorker, Worker

        worker = object.__new__(MemoryBoundWorker)
        with (
            patch.object(worker, "check_memory") as check,
            patch.object(Worker, "execute_model", return_value="answer") as execute,
        ):
            self.assertEqual(worker.execute_model("input"), "answer")
            self.assertEqual(check.call_count, 2)
            execute.assert_called_once_with("input")
