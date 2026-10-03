"""固定 vLLM 0.14.1 的受限显存 Worker；不改动安装包。"""

import os

import torch
from vllm.utils.import_utils import import_pynvml
from vllm.v1.worker.gpu_worker import Worker


class MemoryBoundWorker(Worker):
    def init_device(self):
        super().init_device()
        self.memory_limit = self.vllm_config.additional_config["decider_memory_limit_bytes"]
        # 给 CUDA context、库 workspace 等非 PyTorch 分配保留 3 GB。
        allocator_limit = self.memory_limit - 3_000_000_000
        total = torch.cuda.get_device_properties(self.device).total_memory
        torch.cuda.set_per_process_memory_fraction(allocator_limit / total, self.device)
        self.nvml = import_pynvml()
        self.nvml.nvmlInit()
        self.handles = [
            self.nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.nvml.nvmlDeviceGetCount())
        ]
        self.memory_peak = 0
        self.check_memory()

    def check_memory(self):
        used = sum(
            p.usedGpuMemory
            for h in self.handles
            for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(h)
            if p.pid == os.getpid()
        )
        self.memory_peak = max(self.memory_peak, used)
        if used > self.memory_limit:
            raise RuntimeError(f"decider GPU memory exceeded: {used} > {self.memory_limit}")
        return {
            "used_bytes": used,
            "peak_bytes": self.memory_peak,
            "limit_bytes": self.memory_limit,
        }

    def load_model(self):
        super().load_model()
        self.check_memory()

    def determine_available_memory(self):
        result = super().determine_available_memory()
        self.check_memory()
        return result

    def compile_or_warm_up_model(self):
        super().compile_or_warm_up_model()
        # 释放编译/图捕获遗留的非活跃缓存，避免与首次真实请求的 JIT 临时内存叠加。
        torch.cuda.empty_cache()
        self.check_memory()

    def execute_model(self, scheduler_output):
        self.check_memory()
        result = super().execute_model(scheduler_output)
        self.check_memory()
        return result
