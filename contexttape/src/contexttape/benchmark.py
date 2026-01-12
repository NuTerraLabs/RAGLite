from __future__ import annotations

import csv
import json
import os
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import tiktoken

from .storage import ISStore
from .embed import get_client, embed_text_1536
from .search import combined_search
from .energy import EnergyMeter, EnergyManager


@dataclass
class BenchResult:
    # Corpus stats
    wiki_dir: str
    chat_dir: str
    size_wiki_mb: float
    size_chat_mb: float
    size_total_mb: float
    total_text_segments: int
    total_vector_segments: int
    total_pairs: int
    total_tokens: int
    tokens_per_chunk_mean: float

    # Benchmark config
    queries: int
    repeats: int
    topk: int
    energy_aware: bool

    # Timing & throughput
    wall_ms_p50: float
    wall_ms_p95: float
    wall_ms_mean: float
    qps_mean: float
    elapsed_seconds: float  # total measured time

    # Memory
    rss_mb_max: Optional[float]
    pss_mb_max: Optional[float]

    # Energy
    energy_backend: str
    energy_joules_pkg: Optional[float]
    energy_joules_dram: Optional[float]


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024.0)


def _read_pss_mb_linux() -> Optional[float]:
    """
    Returns proportional set size (PSS) in MB if /proc/self/smaps_rollup exists.
    """
    smaps = "/proc/self/smaps_rollup"
    try:
        with open(smaps, "r") as f:
            for line in f:
                if line.startswith("Pss:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return None


def _corpus_token_stats(stores: List[ISStore], enc) -> Tuple[int, float]:
    total_tokens = 0
    total_chunks = 0
    for store in stores:
        for tid, _ in store.list_pairs():
            text = store.read_text(tid)
            total_tokens += len(enc.encode(text))
            total_chunks += 1
    mean_tokens = (total_tokens / total_chunks) if total_chunks else 0.0
    return total_tokens, mean_tokens


def run_benchmark(
    wiki_dir: str,
    chat_dir: str,
    queries: List[str],
    repeats: int,
    topk: int,
    energy_aware: bool = False,
    max_power_budget: Optional[float] = None
) -> BenchResult:
    wiki = ISStore(wiki_dir)
    chat = ISStore(chat_dir)
    ws = wiki.stat()
    cs = chat.stat()

    # Disk usage
    size_wiki = _dir_size_mb(wiki_dir)
    size_chat = _dir_size_mb(chat_dir)
    size_total = size_wiki + size_chat

    # Token stats
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    total_tokens, mean_tokens = _corpus_token_stats([wiki, chat], enc)

    client = get_client()
    em = EnergyManager()
    meter = EnergyMeter()

    # Pre-embed query vectors
    qvecs = [embed_text_1536(client, q, verbose=False) for q in queries]

    wall_samples: List[float] = []
    total_queries_executed = 0

    proc = psutil.Process(os.getpid()) if psutil else None
    rss_mb_max = 0.0
    pss_mb_max = 0.0

    meter.start()
    t0_all = time.perf_counter()

    for _ in range(repeats):
        for q, qv in zip(queries, qvecs):
            k_use = topk
            stride = 1
            if energy_aware:
                d = em.decide_params(k=topk, max_power_budget=max_power_budget)
                k_use = d["k"]
                stride = d["stride"]

            t0 = time.perf_counter()
            if "stride" in combined_search.__code__.co_varnames:
                _ = combined_search(q, qv, wiki, chat, top_k=k_use, verbose=False, stride=stride)
            else:
                _ = combined_search(q, qv, wiki, chat, top_k=k_use, verbose=False)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            wall_samples.append(dt_ms)
            total_queries_executed += 1

            if proc:
                try:
                    rss = proc.memory_info().rss / (1024 * 1024.0)
                    rss_mb_max = max(rss_mb_max, rss)
                    pss = _read_pss_mb_linux()
                    if pss is not None:
                        pss_mb_max = max(pss_mb_max, pss)
                except Exception:
                    pass

    total_time_s = max(1e-9, time.perf_counter() - t0_all)
    energy = meter.stop()

    # Latency stats
    p50 = statistics.median(wall_samples) if wall_samples else 0.0
    p95 = statistics.quantiles(wall_samples, n=20)[18] if len(wall_samples) >= 20 else max(wall_samples, default=0.0)
    mean = statistics.mean(wall_samples) if wall_samples else 0.0
    qps = total_queries_executed / total_time_s

    return BenchResult(
        wiki_dir=wiki_dir,
        chat_dir=chat_dir,
        size_wiki_mb=size_wiki,
        size_chat_mb=size_chat,
        size_total_mb=size_total,
        total_text_segments=ws["text_segments"] + cs["text_segments"],
        total_vector_segments=ws["vector_segments"] + cs["vector_segments"],
        total_pairs=ws["pairs"] + cs["pairs"],
        total_tokens=total_tokens,
        tokens_per_chunk_mean=mean_tokens,
        queries=len(queries),
        repeats=repeats,
        topk=topk,
        energy_aware=energy_aware,
        wall_ms_p50=p50,
        wall_ms_p95=p95,
        wall_ms_mean=mean,
        qps_mean=qps,
        elapsed_seconds=total_time_s,
        rss_mb_max=(rss_mb_max if rss_mb_max > 0 else None),
        pss_mb_max=(pss_mb_max if pss_mb_max > 0 else None),
        energy_backend=energy.backend,
        energy_joules_pkg=energy.joules_pkg,
        energy_joules_dram=energy.joules_dram,
    )


def save_benchmark(result: BenchResult, out_json: Optional[str], out_csv: Optional[str]) -> None:
    d = asdict(result)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(d.keys())
            w.writerow(d.values())


def render_benchmark_markdown(result: BenchResult) -> str:
    r = result
    return f"""# Retrieval System Benchmark Report

## Corpus & Index Overview
- **Wiki dir:** `{r.wiki_dir}`
- **Chat dir:** `{r.chat_dir}`
- **Disk usage:** total **{r.size_total_mb:.2f} MB** (wiki {r.size_wiki_mb:.2f} MB, chat {r.size_chat_mb:.2f} MB)
- **Segments:** texts **{r.total_text_segments}**, vectors **{r.total_vector_segments}**, pairs **{r.total_pairs}**
- **Tokens:** total **{r.total_tokens:,}**, mean per chunk **{r.tokens_per_chunk_mean:.1f}**

## Benchmark Configuration
- **Queries:** {r.queries}  |  **Repeats:** {r.repeats}  |  **Top-K:** {r.topk}  |  **Energy-aware:** {r.energy_aware}

## Latency & Throughput
- **Latency (ms):** p50 **{r.wall_ms_p50:.2f}**, p95 **{r.wall_ms_p95:.2f}**, mean **{r.wall_ms_mean:.2f}**
- **Throughput:** mean **{r.qps_mean:.2f} QPS** (queries per second)

## Memory
- **RSS max:** {f"{r.rss_mb_max:.2f} MB" if r.rss_mb_max is not None else "unavailable"}
- **PSS max:** {f"{r.pss_mb_max:.2f} MB" if r.pss_mb_max is not None else "unavailable (Linux-only)"}

## Energy
- **Backend:** {r.energy_backend}
- **Package energy:** {f"{r.energy_joules_pkg:.3f} J" if r.energy_joules_pkg is not None else "unavailable"}
- **DRAM energy:** {f"{r.energy_joules_dram:.3f} J" if r.energy_joules_dram is not None else "unavailable"}
- **Elapsed:** {r.elapsed_seconds:.3f} s

---

## Patent Appendix Language (Fill-ins Included)

### Memory Reduction
Preliminary tests on a corpus of **{r.total_text_segments}** text chunks indicated that a conventional vector database (e.g., HNSW-based) required **[INVENTOR INCLUDE – Insert memory usage for conventional DB]** of RAM for its memory-resident index.  
The disclosed file-segment system, when scanning the same corpus (**{r.total_vector_segments}** vectors; **{r.total_tokens:,}** tokens), required less than **{(r.pss_mb_max or r.rss_mb_max or 0):.2f} MB** of RAM during retrieval (peak). This represents a memory reduction of **[INVENTOR INCLUDE – Compute memory reduction percentage]**.

### Energy Consumption
When deployed on **[INVENTOR INCLUDE – hardware platform, e.g., Raspberry Pi 4]**, the system achieved **{r.qps_mean:.2f} QPS** while the measured energy backend was **{r.energy_backend}** and total package energy for the run was **{f"{r.energy_joules_pkg:.3f} J" if r.energy_joules_pkg is not None else "unavailable"}** (elapsed **{r.elapsed_seconds:.2f} s**).  
A conventional vector database failed to run on this hardware or required significantly more power **[INVENTOR INCLUDE – Insert power factor]**.

### Device Applicability
The low memory and compute footprint, specifically the reliance on sequential I/O over in-memory indexes, enables deployment on hardware previously unsuited for retrieval-augmented generation (e.g., in-vehicle infotainment, FPGA-based accelerators, NoC architectures, smartwatches, wearables, and other embedded systems).

### Latency
Full sequential scans over **{r.total_vector_segments}** vectors (≈ **{r.total_tokens:,}** tokens embedded) demonstrated median latency **{r.wall_ms_p50:.2f} ms** and p95 **{r.wall_ms_p95:.2f} ms** for **Top-K={r.topk}** queries under this benchmark. Using hybrid pre-filtering or coarse scanning strategies can further reduce effective latency to **[INVENTOR INCLUDE – Insert reduced latency]** on the same storage medium while preserving accuracy at **[INVENTOR INCLUDE – desired recall@k]**.
"""


def save_benchmark_markdown(md: str, out_md: str) -> None:
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(md, encoding="utf-8")
