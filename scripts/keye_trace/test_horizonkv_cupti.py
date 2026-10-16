"""GPU integration check: nested attribution, real H2D bytes and host idle."""

import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from horizonkv_a6000 import compile_collector


def main():
    with tempfile.TemporaryDirectory(prefix="hkv-cupti-") as tmp:
        root = Path(tmp)
        os.environ["KEYE_HORIZON_KERNEL_DIR"] = str(root)
        os.environ["KEYE_HORIZON_CUPTI_LIBRARY"] = str(compile_collector(root))
        os.environ["KEYE_HORIZON_KERNEL_START"] = "2"
        os.environ["KEYE_HORIZON_KERNEL_STEPS"] = "3"
        import torch
        from sglang.srt.layers.attention.keye_topk.horizonkv_probe import profiled, span

        torch.manual_seed(20261014)
        x = torch.randn(256, 256, device="cuda")
        host = torch.randn(1024, 1024, pin_memory=True)
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True), rids=[], seq_lens_cpu=torch.tensor([4096]))

        class Model:
            @profiled("model_total", batch_arg=1)
            def forward(self, batch):
                with span(batch, 0, "indexer_score"):
                    y = x @ x
                    time.sleep(.01)
                with span(batch, 0, "attention_compute"):
                    y = y.softmax(-1)
                with span(batch, 0, "kv_h2d", bytes=host.nbytes):
                    z = host.to("cuda", non_blocking=True)
                return y, z

        model = Model()
        for wave in range(2):
            batch.rids = [f"kernel__smoke__wave{wave:03d}__case"]
            for _ in range(7):
                model.forward(batch)
        paths = sorted(root.glob("*.summary.json"))
        assert len(paths) == 2
        for path in paths:
            value = json.loads(path.read_text())
            assert value["captured_steps"] == 3 and value["cuda_activity_sum_us"] > 0
            assert value["collector_metadata"]["dropped"] == 0
            assert value["collector_metadata"]["unattributed"] == 0
            ranges = value["ranges"]
            score_gpu = sum(r["gpu_activity_us"] for r in ranges if r["component"] == "indexer_score")
            score_cpu = sum(r["cpu_range_exclusive_us"] for r in ranges if r["component"] == "indexer_score")
            assert 0 < score_gpu < score_cpu / 2, (score_gpu, score_cpu)
            assert abs(sum(r["gpu_activity_us"] for r in ranges) - value["cuda_activity_sum_us"]) < .01
            activities = [json.loads(line) for line in path.with_name(path.name.replace(".summary.json", ".activities.jsonl")).read_text().splitlines()]
            copies = [r for r in activities if r["kind"] == "memcpy" and r["copy_kind"] == 1]
            assert sum(r["bytes"] for r in copies) == 3 * host.nbytes
        print("PASS: two captures; nested GPU attribution conserves activity sum; zero drops/unattributed; exact H2D bytes; CPU idle excluded")


if __name__ == "__main__":
    main()
