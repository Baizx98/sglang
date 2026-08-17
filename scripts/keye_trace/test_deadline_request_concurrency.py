#!/usr/bin/env python3
"""Integration test for concurrent deadline request orchestration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class MockHandler(BaseHTTPRequestHandler):
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, value: object) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health_generate":
            self._send_json({"healthy": True})
        elif self.path == "/model_info":
            self._send_json({"model_path": "mock"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/flush_cache":
            self._send_json({"flushed": True})
            return
        if self.path != "/generate":
            self.send_error(404)
            return
        with self.lock:
            type(self).active += 1
            type(self).maximum_active = max(
                type(self).maximum_active, type(self).active
            )
        time.sleep(0.08)
        with self.lock:
            type(self).active -= 1
        self._send_json(
            {
                "text": "mock output",
                "output_ids": [1, 2],
                "meta_info": {
                    "completion_tokens": 2,
                    "e2e_latency": 0.08,
                    "rid": payload.get("rid"),
                },
            }
        )


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared.jsonl"
            with prepared.open("w") as handle:
                for index in range(5):
                    handle.write(
                        json.dumps(
                            {
                                "task": f"task-{index}",
                                "rid": f"source-{index}",
                                "prompt_len": 3,
                                "input_ids": [1, 2, 3],
                            }
                        )
                        + "\n"
                    )
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/keye_trace/run_deadline_trace_requests.py",
                    "--output-dir",
                    str(output),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--prepared-requests",
                    str(prepared),
                    "--request-indices",
                    "0,2,3,4",
                    "--rid-prefix",
                    "concurrency-test",
                    "--concurrency",
                    "4",
                    "--max-new-tokens",
                    "2",
                    "--min-new-tokens",
                    "2",
                    "--flush-cache",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads((output / "summary.json").read_text())
            manifest = json.loads((output / "wave_manifest.json").read_text())
            requests = [
                json.loads(line)
                for line in (output / "requests.jsonl").read_text().splitlines()
            ]
            assert summary["requests"] == 4
            assert summary["requested_concurrency"] == 4
            assert summary["waves"] == 1
            assert summary["all_structural_passed"]
            assert len(manifest["waves"][0]["request_ids"]) == 4
            assert len(requests) == 4
            assert {row["request_index"] for row in requests} == {0, 2, 3, 4}
            assert MockHandler.maximum_active == 4
            print(
                json.dumps(
                    {
                        "passed": True,
                        "maximum_active_requests": MockHandler.maximum_active,
                        "start_skew_ms": summary[
                            "maximum_request_start_skew_ms"
                        ],
                    }
                )
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
