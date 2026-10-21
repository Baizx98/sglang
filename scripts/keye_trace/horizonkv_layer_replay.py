"""Real Keye layer replay on one A6000, independent of full-model residency."""

import argparse
import json
import os
import socket
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KEYE_SM80_DSA", "1")
os.environ.setdefault("KEYE_SM80_EXACT_TOPK", "1")

import torch
from safetensors import safe_open
from transformers import PretrainedConfig

MODEL = Path("/Tan/model/Keye-VL-2.0-30B-A3B")


def initialize():
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.layers.moe.utils import initialize_moe_config

    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    args = ServerArgs(model_path=str(MODEL), device="cuda", tp_size=1, trust_remote_code=True,
                      disable_custom_all_reduce=True, disable_cuda_graph=True,
                      moe_runner_backend="triton")
    set_global_server_args_for_scheduler(args)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                distributed_init_method=f"tcp://127.0.0.1:{port}")
    initialize_model_parallel(tensor_model_parallel_size=1)
    initialize_dp_attention(args, SimpleNamespace(hidden_size=2048, dtype=torch.bfloat16))
    initialize_moe_config(args)
    from sglang.srt.utils.hf_transformers.config import get_config
    return get_config(str(MODEL), trust_remote_code=True)


def load_layer(config, layer_id):
    from sglang.srt.models.keye_topk_mask_moe import (
        KeyeTopKMaskMoeDecoderLayer, KeyeVL2MoeForConditionalGeneration)

    with torch.device("cuda"):
        layer = KeyeTopKMaskMoeDecoderLayer(config, layer_id=layer_id)
    prefix = f"model.layers.{layer_id}."
    owner = SimpleNamespace(config=config, _cached_params_dict={
        prefix + name: param for name, param in layer.named_parameters()})
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    names = [name for name in index if name.startswith(prefix)]

    def weights():
        for name in names:
            with safe_open(str(MODEL / index[name]), framework="pt", device="cpu") as stream:
                yield name, stream.get_tensor(name)

    KeyeVL2MoeForConditionalGeneration.load_weights(owner, weights())
    for module in layer.modules():
        method = getattr(module, "quant_method", None)
        if method is not None and hasattr(method, "process_weights_after_loading"):
            method.process_weights_after_loading(module)
    layer.eval()
    return layer


def read_frames(root, wave, layer, batch=None):
    fixtures = Path(root) / "fixtures"
    records = [json.loads(line) for line in (fixtures / "manifest_rank00.jsonl").read_text().splitlines()]
    frames = []
    for record in records:
        if (record["layer"] == layer and record["request_ids"][0].startswith(f"replay__wave{wave:03d}__")
                and (batch is None or len(record["request_ids"]) == batch)):
            value = torch.load(fixtures / record["file"], weights_only=False)
            peer = torch.load(fixtures / record["file"].replace("rank00", "rank01"), weights_only=False)
            value["peer"] = peer
            frames.append(value)
    return frames


class ReplayData:
    """Logical request histories reconstructed from real initial KV and deltas."""

    def __init__(self, root, frames, frame):
        self.frame = frame
        self.batch = len(frame["request_ids"])
        self.lengths = frame["seq_lens"].int().cuda()
        self.maxlen = int(frame["seq_lens"].max())
        self.prior = {rid: [] for rid in frame["request_ids"]}
        self.prior_inputs = {rid: [] for rid in frame["request_ids"]}
        self.hostk = torch.zeros(self.batch, self.maxlen, 4, 128, dtype=torch.bfloat16).pin_memory()
        self.hostv = torch.zeros_like(self.hostk).pin_memory()
        index = torch.zeros(self.batch, self.maxlen, 64, dtype=torch.bfloat16)
        root = Path(root) / "fixtures"
        for row, rid in enumerate(frame["request_ids"]):
            prefix = f"{rid}_layer{frame['layer']:02d}_rank"
            a = torch.load(root / f"{prefix}00_history.pt", weights_only=False)
            b = torch.load(root / f"{prefix}01_history.pt", weights_only=False)
            length = a["length"]
            self.hostk[row, :length] = torch.cat((a["main_k"], b["main_k"]), dim=1)
            self.hostv[row, :length] = torch.cat((a["main_v"], b["main_v"]), dim=1)
            index[row, :length] = a["index_k"]
            current_step = frame["steps"][row]
            for f in frames:
                if rid not in f["request_ids"]:
                    continue
                i = f["request_ids"].index(rid)
                if f["steps"][i] > current_step:
                    continue
                pos = int(f["seq_lens"][i]) - 1
                if pos >= self.maxlen:
                    continue
                d, p = f["deltas"][i], f["peer"]["deltas"][i]
                self.hostk[row, pos] = torch.cat((d["main_k"], p["main_k"]), dim=0)
                self.hostv[row, pos] = torch.cat((d["main_v"], p["main_v"]), dim=0)
                index[row, pos] = d["index_k"]
                if f["steps"][i] < current_step:
                    self.prior[rid].append((f["steps"][i], f["topk"][i]))
                    self.prior_inputs[rid].append((f["steps"][i], f["attention_input"][i:i+1],
                                                  f["positions"][..., i:i+1], int(f["seq_lens"][i])))
        self.index = index.view(-1, 64).cuda()
        self.page_table = torch.arange(self.batch * self.maxlen, device="cuda", dtype=torch.int32).reshape(self.batch, -1)
        self.current = (self.page_table[:, 0] + self.lengths - 1).long()
        self.positions = frame["positions"].cuda()
        self.hidden = frame["hidden"].cuda()
        self.residual = None if frame["residual"] is None else frame["residual"].cuda()
        if frame["needs_allreduce"]:
            self.hidden = (frame["hidden"] + frame["peer"]["hidden"]).cuda()
        self.source_attention_input = frame["attention_input"].cuda()
        self.source_query = torch.cat((frame["query"], frame["peer"]["query"]), dim=1).reshape(self.batch, -1).cuda()
        self.source_attention_output = torch.cat((frame["attention_output"], frame["peer"]["attention_output"]), dim=-1).cuda()


class ReplayLayer:
    def __init__(self, layer, data, mode, capacity=4096):
        from horizonkv_host_kernels import DeviceLRU, HostPointer
        from sglang.srt.layers.attention.keye_dsa_sm80 import KeyeDSAWorkspace

        self.layer, self.data, self.mode = layer, data, mode
        self.batch, self.topk = data.batch, layer.self_attn.sa_topk
        self.hostk, self.hostv = HostPointer(data.hostk), HostPointer(data.hostv)
        self.k = self.v = None
        if mode in {"h-native", "h-staged"}:
            self.k = data.hostk.view(-1, 4, 128).cuda()
            self.v = data.hostv.view(-1, 4, 128).cuda()
        self.stagek = torch.empty(self.batch, self.topk, 4, 128, device="cuda", dtype=torch.bfloat16)
        self.stagev = torch.empty_like(self.stagek)
        self.stage_slots = torch.arange(self.batch * self.topk, device="cuda", dtype=torch.int32).reshape(self.batch, -1)
        self.cache = None
        if mode in {"l-cached", "no-io", "ahead", "query-ready", "echo", "topk-ready"}:
            self.cache = DeviceLRU(self.batch, data.maxlen, capacity, self.topk)
            self.prime_cache()
        self.prefmap = torch.full((self.batch, data.maxlen), -1, device="cuda", dtype=torch.int32)
        self.prefk = torch.empty_like(self.stagek)
        self.prefv = torch.empty_like(self.stagev)
        self.prefcounts = torch.zeros(self.batch, device="cuda", dtype=torch.int32)
        self.workspace = KeyeDSAWorkspace()
        self.record_stages = False
        self.stage_events = []
        self.prefetch_done = None
        self.prefetch_stream = None
        self.fixed_topk = None
        self.threshold = torch.zeros(self.batch, device="cuda", dtype=torch.float32)

    @torch.inference_mode()
    def prepare_echo(self):
        from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged
        indexer = self.layer.self_attn.sa_indexer
        for row, rid in enumerate(self.data.frame["request_ids"]):
            ema = None
            for _, h, pos, length in sorted(self.data.prior_inputs[rid], key=lambda v: v[0]):
                q, _, gate = indexer._get_q_k_w_bf16(h.cuda(), pos.cuda(), False)
                lens = torch.tensor([length], device="cuda", dtype=torch.int32)
                score = keye_indexer_score_paged(q, self.data.index,
                                                self.data.page_table[row:row+1], gate,
                                                lens, indexer.softmax_scale)
                cutoff = score.topk(min(self.topk, length), dim=-1).values[:, -1]
                ema = cutoff if ema is None else 0.5 * ema + 0.5 * cutoff
            if ema is not None:
                self.threshold[row] = ema[0]

    def prime_cache(self):
        cache = self.cache
        for row, rid in enumerate(self.data.frame["request_ids"]):
            order = OrderedDict()
            for _, indices in sorted(self.data.prior[rid], key=lambda v: v[0]):
                wanted = [int(v) for v in indices.tolist() if v >= 0]
                hits = [v for v in wanted if v in order]
                misses = [v for v in wanted if v not in order]
                for v in hits:
                    order.move_to_end(v)
                for v in misses:
                    if len(order) == cache.capacity:
                        order.popitem(last=False)
                    order[v] = None
                for v in hits:
                    order.move_to_end(v)
            tags = list(order)
            if tags:
                ids = torch.tensor(tags, device="cuda", dtype=torch.long)
                cache.map[row, ids] = torch.arange(len(tags), device="cuda", dtype=torch.int32)
                cache.tag[row, :len(tags)] = ids.int()
                cache.age[row, :len(tags)] = torch.arange(len(tags), device="cuda", dtype=torch.int32)
                cache.k[row, :len(tags)] = self.data.hostk[row, tags].cuda()
                cache.v[row, :len(tags)] = self.data.hostv[row, tags].cuda()
        self.initial_cache = {name: getattr(cache, name).clone() for name in ("map", "tag", "age", "k", "v")}

    def reset(self):
        if not hasattr(self, "hidden"):
            self.hidden = self.data.hidden.clone()
            self.residual = None if self.data.residual is None else self.data.residual.clone()
        else:
            self.hidden.copy_(self.data.hidden)
            if self.residual is not None:
                self.residual.copy_(self.data.residual)
        if self.cache is not None:
            for name, value in self.initial_cache.items():
                getattr(self.cache, name).copy_(value)

    @contextmanager
    def span(self, name):
        if not self.record_stages:
            yield
            return
        start, end = (torch.cuda.Event(enable_timing=True, external=True),
                      torch.cuda.Event(enable_timing=True, external=True))
        start.record()
        with torch.profiler.record_function(f"layer{self.layer.layer_id}/{name}"):
            yield
        end.record()
        self.stage_events.append((name, start, end))

    def forward(self, hidden=None, residual=None):
        from horizonkv_host_kernels import gather, write_current, update_prefetched_current
        from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged
        from sglang.srt.layers.attention.keye_dsa_sm80 import keye_dsa_paged_attention
        from sglang.srt.layers.attention.nsa.transform_index import transform_index_page_table_decode

        h = self.hidden if hidden is None else hidden
        residual = self.residual if hidden is None else residual
        layer, data, attn = self.layer, self.data, self.layer.self_attn
        with self.span("input_norm"):
            if residual is None:
                residual = h
                h = layer.input_layernorm(h)
            else:
                h, residual = layer.input_layernorm(h, residual)
        self.last_attention_input = h
        with self.span("qkv_projection"):
            qkv, _ = attn.qkv_proj(h)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        with self.span("qk_norm_rope"):
            q, k = attn._apply_qk_norm(q, k)
            q, k = attn.rotary_emb(data.positions, q, k)
        self.last_query = q
        with self.span("indexer_query_prepare"):
            iq, ik, weights = attn.sa_indexer._get_q_k_w_bf16(h, data.positions, False)
            data.index.index_copy_(0, data.current, ik)
        with self.span("indexer_score"):
            if self.fixed_topk is not None:
                scores = None
            elif self.mode == "echo":
                from horizonkv_host_kernels import score_prefetch
                scores = score_prefetch(iq, data.index, data.page_table, weights,
                                        data.lengths, attn.sa_indexer.softmax_scale,
                                        self.threshold, self.hostk, self.hostv,
                                        self.cache.map, self.prefmap, self.prefk,
                                        self.prefv, self.prefcounts)
            else:
                scores = keye_indexer_score_paged(iq, data.index, data.page_table,
                                                 weights, data.lengths, attn.sa_indexer.softmax_scale)
        with self.span("indexer_topk"):
            indices = self.fixed_topk if self.fixed_topk is not None else attn.sa_indexer._topk_sm80(scores, data.lengths)
        self.last_topk = indices
        with self.span("kv_writeback"):
            targetk, targetv = (self.k, self.v) if self.k is not None else (self.hostk, self.hostv)
            write_current(targetk, targetv, k, v, data.lengths, data.maxlen)
            if self.mode == "no-io":
                update_prefetched_current(self.prefmap, self.prefk, self.prefv,
                                           k, v, data.lengths, data.maxlen, self.topk)
        with self.span("kv_resolve_gather"):
            if self.mode == "h-native":
                slots = transform_index_page_table_decode(page_table=data.page_table,
                                                          topk_indices=indices, page_size=1)
                cachek, cachev = self.k, self.v
            else:
                if self.prefetch_stream is not None:
                    # An internal wait joins the fork into the capture graph.
                    # External timing events alone do not close stream capture.
                    torch.cuda.current_stream().wait_stream(self.prefetch_stream)
                if self.cache is None:
                    sourcek, sourcev = (self.k, self.v) if self.k is not None else (self.hostk, self.hostv)
                    gather(sourcek, sourcev, indices, self.stagek, self.stagev, data.lengths, data.maxlen)
                else:
                    pref = (self.prefmap, self.prefk, self.prefv) if self.mode in {"no-io", "ahead", "query-ready", "echo"} else None
                    self.cache.resolve(self.hostk, self.hostv, indices, self.stagek, self.stagev,
                                       data.lengths, max(data.frame["steps"]) + 1, pref)
                slots = self.stage_slots
                if int(data.frame["seq_lens"].min()) < self.topk:
                    slots = slots.masked_fill(indices < 0, -1)
                cachek = self.stagek.view(-1, 4, 128)
                cachev = self.stagev.view(-1, 4, 128)
        with self.span("attention"):
            output = keye_dsa_paged_attention(q.view(self.batch, 32, 128), cachek,
                                              cachev, slots, num_q_heads=32, head_dim=128,
                                              sm_scale=attn.scaling, is_decode=True,
                                              workspace=self.workspace)
        self.last_attention_output = output
        with self.span("output_projection"):
            h, _ = attn.o_proj(output)
        with self.span("post_attention_norm"):
            h, residual = layer.post_attention_layernorm(h, residual)
        with self.span("moe_ffn"):
            h = layer.mlp(h)
        return h, residual

    def prepare_no_io(self, truth):
        indices = truth.cpu().long()
        for row in range(self.batch):
            valid = indices[row] >= 0
            ids = indices[row, valid]
            self.prefk[row, valid.cuda()] = self.data.hostk[row, ids].cuda()
            self.prefv[row, valid.cuda()] = self.data.hostv[row, ids].cuda()
            self.prefmap[row, ids.cuda()] = torch.arange(self.topk, device="cuda", dtype=torch.int32)[valid.cuda()]


@torch.inference_mode()
def graph_measure(call, reset, runners, repeats=5, profile_path=None, transfer_events=None):
    """Measure uninstrumented graphs; probe events and Kineto run separately."""
    for runner in runners:
        runner.record_stages = False
    for _ in range(3):
        reset()
        call()
    torch.cuda.synchronize()
    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    for _ in range(25):
        reset()
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        reset()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    expected = tuple(x.clone() for x in output)
    # Event nodes preserve the actual multi-stream dependency timeline.
    for runner in runners:
        runner.record_stages = True
        runner.stage_events = []
    reset()
    probe = torch.cuda.CUDAGraph()
    with torch.cuda.graph(probe):
        call()
    reset()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    probe.replay()
    b.record()
    b.synchronize()
    spans = []
    transfer = None
    if transfer_events is not None:
        transfer = {"start_us": a.elapsed_time(transfer_events[0]) * 1000,
                    "end_us": a.elapsed_time(transfer_events[1]) * 1000,
                    "stream_span_us": transfer_events[0].elapsed_time(transfer_events[1]) * 1000}
    for runner in runners:
        spans.extend({"layer": runner.layer.layer_id, "component": name,
                      "start_us": a.elapsed_time(start) * 1000,
                      "end_us": a.elapsed_time(end) * 1000,
                      "stream_span_us": start.elapsed_time(end) * 1000}
                     for name, start, end in runner.stage_events)
        runner.record_stages = False
    work = {}
    if profile_path is not None:
        for runner in runners:
            runner.record_stages = True
            runner.stage_events = []
        reset()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(profile_path))
        from horizonkv_trace_work import kernel_work_from_trace
        work = kernel_work_from_trace(profile_path)["component_gpu_work_us"]
        if not sum(work.values()):
            raise RuntimeError("CUPTI captured no GPU activities; profile must be repeated")
        for runner in runners:
            runner.record_stages = False
    return {"elapsed_us": times, "probe_elapsed_us": a.elapsed_time(b) * 1000,
            "probe_component_spans": spans, "prefetch_stream_span": transfer,
            "profile_component_gpu_work_us": work}, expected


@torch.inference_mode()
def pair_call(previous, target, mode, truth=None):
    """Build a callable with an identical two-layer measurement boundary."""
    from horizonkv_host_kernels import prefetch

    stream = torch.cuda.Stream()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    if mode in {"ahead", "query-ready"}:
        target.prefetch_done = end
        target.prefetch_stream = stream

    def transfer():
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            begin.record()
            prefetch(target.hostk, target.hostv, truth, target.cache.map,
                     target.prefmap, target.prefk, target.prefv,
                     target.data.lengths, target.prefcounts, target.data.maxlen)
            end.record()

    def run():
        if mode == "ahead":
            transfer()
        h, residual = previous.forward()
        if mode == "query-ready":
            transfer()
        return target.forward(h, residual)

    def reset():
        previous.reset()
        target.reset()

    return run, reset, (begin, end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-smoke", type=int)
    parser.add_argument("--validate-wave", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--graph-check", action="store_true")
    args = parser.parse_args()
    config = initialize()
    if args.weights_smoke is not None:
        layer = load_layer(config, args.weights_smoke)
        with torch.inference_mode():
            output = layer.mlp(torch.ones(4, config.hidden_size, device="cuda"))
        torch.cuda.synchronize()
        print(json.dumps({"layer": args.weights_smoke, "gpu": torch.cuda.get_device_name(),
                          "parameter_bytes": sum(p.numel() * p.element_size() for p in layer.parameters()),
                          "moe_output_shape": list(output.shape), "finite": bool(output.isfinite().all())}))
    if args.validate_wave is not None:
        frames = read_frames(args.run_dir, args.validate_wave, args.layer)
        frame = next(f for f in frames if len(f["request_ids"]) == 4)
        data = ReplayData(args.run_dir, frames, frame)
        layer = load_layer(config, args.layer)
        baseline = None
        for mode in ("h-native", "h-staged", "d-staged", "l-cached", "no-io", "echo"):
            runner = ReplayLayer(layer, data, mode)
            if mode == "no-io":
                runner.prepare_no_io(truth)
            if mode == "echo":
                runner.prepare_echo()
            runner.reset()
            with torch.inference_mode():
                value, residual = runner.forward()
            torch.cuda.synchronize()
            if baseline is None:
                baseline, baseline_residual = value.clone(), residual.clone()
                truth = runner.last_topk.clone()
            record = {"mode": mode, "layer": args.layer, "batch": data.batch,
                      "lengths": data.lengths.cpu().tolist(),
                      "normalization_max_error": float((runner.last_attention_input - data.source_attention_input).abs().max()),
                      "query_max_error": float((runner.last_query - data.source_query).abs().max()),
                      "attention_max_error": float((runner.last_attention_output - data.source_attention_output).abs().max()),
                      "topk_set_matches_source": bool(torch.equal(truth.sort(dim=-1).values.cpu(), frame["topk"].sort(dim=-1).values)),
                      "topk_matches_baseline": bool(torch.equal(truth, runner.last_topk)),
                      "output_matches_baseline": bool(torch.equal(value, baseline) and torch.equal(residual, baseline_residual))}
            print(json.dumps(record), flush=True)
            assert record["topk_matches_baseline"] and record["output_matches_baseline"]
            if args.graph_check:
                _, check = graph_measure(runner.forward, runner.reset, [runner], repeats=1)
                assert torch.equal(check[0], baseline) and torch.equal(check[1], baseline_residual)
                print(json.dumps({"mode": mode, "graph_correctness": True}), flush=True)
            del runner
    torch.distributed.destroy_process_group()
