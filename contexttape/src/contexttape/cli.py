# contexttape/cli.py
from __future__ import annotations

import argparse
from argparse import RawDescriptionHelpFormatter
from pathlib import Path
import json
import os
import hashlib
from typing import List, Optional, Iterable, Tuple, Set

from .config import WIKI_IS_DIR, CHAT_IS_DIR, TOP_K_DEFAULT, DEBUG_DIR
from .utils import ensure_dir
from .storage import ISStore, DT_TEXT
from .embed import get_client, embed_text_1536
from .ingest_wiki import load_topics, fetch_wiki_page
from .ingest_generic import iter_files
from .search import combined_search
from .chat import store_chat_turn
from .ingest_any import ingest_any

from .benchmark import (
    run_benchmark,
    save_benchmark,
    render_benchmark_markdown,
    save_benchmark_markdown,
)
from .energy import EnergyManager
from .relevance import select_relevant_blocks

import tiktoken

VERSION = "0.5.0"

def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

HELP_EPILOG = f"""\
Environment variables
---------------------
  OPENAI_API_KEY   Required for embeddings/chat (e.g., "sk-...").
  WIKI_IS_DIR      Default wiki store directory (default: {WIKI_IS_DIR}).
  CHAT_IS_DIR      Default chat store directory (default: {CHAT_IS_DIR}).
  TOP_K            Default top-k (default: {TOP_K_DEFAULT}).
  DEBUG_DIR        Where --verbose logs go (default: {DEBUG_DIR}).
  CTX_SEG_EXT      Optional custom segment file extension (e.g. ".ismail").

Quick start
-----------
  # 1) Install (venv recommended)
  pip install -e .

  # 2) Set API key for embeddings/chat
  export OPENAI_API_KEY="sk-..."

  # 3) Seed a small multimodal corpus then ingest
  python scripts/seed_multimodal_corpus.py --out sample_corpus
  ct ingest-any sample_corpus --out-dir wiki_store --quantize --verbose

  # 4) Chat with retrieval
  ct chat --wiki-dir data/wiki --chat-dir data/chat --topk 8 --alpha 0.55 \
          --min-lex 0.15 --min-hybrid 0.28 --max-context-blocks 5 --verbose

Tip: Run "ct <command> --help" for command-specific examples and templates.
"""

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _resolve_dirs(args):
    wiki_dir = (
        getattr(args, "wiki_dir", None)
        or getattr(args, "wiki_store", None)
        or WIKI_IS_DIR
    )
    chat_dir = (
        getattr(args, "chat_dir", None)
        or getattr(args, "chat_ts", None)
        or CHAT_IS_DIR
    )
    return wiki_dir, chat_dir


def _save_debug(hits, blocks: List[str], prompt: str) -> None:
    ensure_dir(DEBUG_DIR)
    base = Path(DEBUG_DIR)
    safe_hits = [
        [str(src), float(score), int(tid), int(eid)] for (src, score, tid, eid) in hits
    ]
    (base / "hits.json").write_text(json.dumps(safe_hits, indent=2), encoding="utf-8")
    (base / "context.md").write_text("\n\n---\n\n".join(blocks), encoding="utf-8")
    (base / "prompt.txt").write_text(prompt, encoding="utf-8")


def _supports(fn, *names: str) -> dict:
    varnames = fn.__code__.co_varnames
    return {n: (n in varnames) for n in names}


def _call_combined_search(q, qvec, wiki_store, chat_store, **maybe_kwargs):
    sup = _supports(
        combined_search,
        "top_k",
        "verbose",
        "stride",
        "alpha",
        "min_vec",
        "min_lex",
        "min_hybrid",
        "use_shards",
    )
    kwargs = {}
    for k, v in maybe_kwargs.items():
        if sup.get(k, False):
            kwargs[k] = v
    return combined_search(q, qvec, wiki_store, chat_store, **kwargs)


def _call_select_relevant_blocks(user, hits, wiki_store, chat_store, **maybe_kwargs):
    sup = _supports(
        select_relevant_blocks,
        "topk",
        "max_blocks",
        "max_preview_chars",
        "alpha",
        "min_vec",
        "min_lex",
        "min_hybrid",
    )
    kwargs = {}
    for k, v in maybe_kwargs.items():
        if sup.get(k, False):
            kwargs[k] = v
    return select_relevant_blocks(user, hits, wiki_store, chat_store, **kwargs)


def _hash_text(s: str) -> str:
    import hashlib as _hl
    return _hl.sha1(s.strip().encode("utf-8")).hexdigest()


def _dedupe_blocks(blocks: List[str]) -> List[str]:
    seen = set()
    out = []
    for b in blocks:
        h = _hash_text(b)
        if h not in seen:
            seen.add(h)
            out.append(b)
    return out


def _is_assistant_line(line: str) -> bool:
    return line.lower().startswith("assistant:")


def _first_payload_line(block: str) -> Optional[str]:
    for ln in block.splitlines():
        if ":" in ln and (ln.lower().startswith("user:") or ln.lower().startswith("assistant:")):
            return ln.strip()
    return None


def _label_blocks(blocks: Iterable[str], label: str) -> List[str]:
    labeled = []
    for i, b in enumerate(blocks, 1):
        labeled.append(f"[{label} {i}]\n{b}")
    return labeled


def _read_recent_chat_window(store: ISStore, window: int = 8) -> Tuple[List[str], Set[int]]:
    tids = sorted(store.list_segments(DT_TEXT))
    if not tids:
        return [], set()
    take = tids[-window:]
    tidset = set(take)
    blocks = []
    for tid in take:
        try:
            txt = store.read_text(tid)
        except Exception:
            continue
        blocks.append(txt.strip())
    return blocks, tidset


def _filter_knowledge_blocks(blocks: List[str], prefer_assistant: bool = True) -> List[str]:
    out: List[str] = []
    for b in blocks:
        first = _first_payload_line(b)
        if first is None or (not first.startswith("user:") and not first.startswith("assistant:")):
            out.append(b)
            continue
        if prefer_assistant and not _is_assistant_line(first):
            s = first.split(":", 1)[-1].strip()
            if len(s) < 60 or s.lower() in {"hi", "hello", "hey", "what’s up", "whats up"}:
                continue
        out.append(b)
    return out


def _extract_tid_from_block(block: str) -> Optional[int]:
    for ln in block.splitlines():
        if "text_path=" in ln and "segment_" in ln and ".is" in ln:
            part = ln.split("segment_")[-1]
            try:
                return int(part.split(".is")[0])
            except Exception:
                return None
    return None


def _block_source(block: str, wiki_dir: str, chat_dir: str) -> str:
    wiki_base = Path(wiki_dir).name
    chat_base = Path(chat_dir).name
    for ln in block.splitlines():
        if "text_path=" in ln:
            if wiki_base in ln:
                return "WIKI"
            if chat_base in ln:
                return "CHAT"
    header = block.splitlines()[0].upper() if block.splitlines() else ""
    if header.startswith("[WIKI"):
        return "WIKI"
    return "CHAT"


def _search_store_only(
    store: ISStore,
    qvec,
    top_k: int = 5,
    stride: int = 1,
    coarse_limit: Optional[int] = 16,
):
    return store.search_by_vector(
        query_vec=qvec, top_k=top_k, stride=stride, coarse_limit=coarse_limit
    )


def _format_preview(text: str, n: int = 220) -> str:
    t = (text or "").replace("\n", " ")
    return (t[:n] + "...") if len(t) > n else t


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def cmd_build_wiki(args) -> None:
    topics = load_topics(args.topics_file, args.limit)
    if args.skip_fences:
        topics = [t for t in topics if t and not str(t).strip().startswith("```")]
    topics = [t.strip() for t in topics if str(t).strip()]

    out_dir = args.out_dir or WIKI_IS_DIR
    wiki_store = ISStore(out_dir)
    client = get_client()

    total_tokens_exact = 0

    for i, topic in enumerate(topics, 1):
        if args.verbose:
            print(f"[{i:02d}/{len(topics)}] {topic}")

        text = fetch_wiki_page(topic, verbose=args.verbose)

        if args.min_chars and len(text) < args.min_chars:
            if args.verbose:
                print(f"  skipped (len<{args.min_chars})")
            continue

        # exact model tokens for the FULL page text
        n_tok = count_tokens(text)
        total_tokens_exact += n_tok
        if args.verbose:
            print(f"[TOKENS] model=text-embedding-3-small tokens={n_tok}")

        emb = embed_text_1536(client, text, verbose=args.verbose)
        t_id, e_id = wiki_store.append_text_with_embedding(text, emb)
        if args.verbose:
            print(f"  wrote text={t_id} vec={e_id}")

    stats = wiki_store.stat()
    print(
        f"[DONE] dir={wiki_store.dir_path} "
        f"segments={stats['text_segments'] + stats['vector_segments']} "
        f"pairs={stats['pairs']} tokens={total_tokens_exact}"
    )
    print(f"\n✅ Wikipedia knowledge base ready!")
    print(f"   Try: ct search \"your question\"")
    print(f"   Or:  ct chat\n")


def cmd_ingest_path(args) -> None:
    out_dir = args.out_dir or WIKI_IS_DIR
    store = ISStore(out_dir)
    client = get_client()

    exts: Optional[List[str]] = None
    if args.exts:
        exts = [e if e.startswith(".") else f".{e}" for e in args.exts]

    if args.verbose:
        print(f"[INFO] ingest out_dir={out_dir} exts={exts or 'ALL'}")

    total = 0
    for path, text in iter_files(
        root=args.path,
        exts=exts,
        max_bytes=args.max_bytes,
        max_pdf_pages=args.max_pdf_pages,
        follow_symlinks=args.follow_symlinks,
    ):
        if args.verbose:
            print(f"[READ] {path}")
        emb = embed_text_1536(
            client,
            text,
            chunk_tokens=args.embed_chunk_tokens,
            chunk_overlap=args.embed_chunk_overlap,
            pool=args.embed_pool,
            verbose=False,
        )
        store.append_text_with_embedding(text, emb)
        total += 1

    stats = store.stat()
    print(
        f"[DONE] dir={store.dir_path} files_ingested={total} pairs={stats['pairs']} "
        f"texts={stats['text_segments']} vecs={stats['vector_segments']}"
    )


def cmd_search(args) -> None:
    client = get_client()
    q = " ".join(args.query)
    # Query text is small; chunk flags are harmless but unused in one-shot
    qvec = embed_text_1536(
        client,
        q,
        chunk_tokens=args.embed_chunk_tokens,
        chunk_overlap=args.embed_chunk_overlap,
        pool=args.embed_pool,
        verbose=args.verbose,
    )

    wiki_dir, chat_dir = _resolve_dirs(args)
    if args.verbose:
        print(f"[INFO] wiki_dir={wiki_dir} chat_dir={chat_dir}")

    wiki_store = ISStore(wiki_dir)
    chat_store = ISStore(chat_dir)

    if not (wiki_store.list_pairs() or chat_store.list_pairs()):
        print(f"\n[ERROR] No data found to search!")
        print(f"  wiki_dir: {wiki_dir} (empty)")
        print(f"  chat_dir: {chat_dir} (empty)")
        print(f"\nTo get started:")
        print(f"  1. Create topics.txt with Wikipedia page titles")
        print(f"  2. Run: ct build-wiki --topics-file topics.txt --limit 3")
        print(f"  3. Then search: ct search \"your query\"\n")
        return

    k_use = args.topk
    stride = 1
    if args.energy_aware:
        policy = EnergyManager()
        d = policy.decide_params(k=args.topk, max_power_budget=args.max_power_budget)
        k_use = d["k"]
        stride = d["stride"]
        if args.verbose:
            print(f"[ENERGY] adapted topk={k_use} stride={stride} quant={d['quant_level']}")

    wiki_hits_only = _search_store_only(
        wiki_store, qvec, top_k=(args.wiki_topk or k_use), stride=stride, coarse_limit=16
    )
    chat_hits_only = _search_store_only(
        chat_store, qvec, top_k=(args.chat_topk or k_use), stride=stride, coarse_limit=16
    )

    if not wiki_hits_only and not chat_hits_only:
        print("No hits in either store.")
        return

    print("\n=== WIKI RESULTS ===")
    if wiki_hits_only:
        for rank, (score, tid, eid) in enumerate(wiki_hits_only, 1):
            text = wiki_store.read_text(tid)
            preview = _format_preview(text, 220)
            print(f"[TOP {rank}] src=wiki score={score:.4f} text_seg={tid} emb_seg={eid}")
            print(f"  text_path={wiki_dir}/segment_{tid}.is")
            print(f"  vec_path={wiki_dir}/segment_{eid}.is")
            print(f"  preview: {preview}\n")
    else:
        print("(no wiki hits)")

    print("\n=== CHAT RESULTS ===")
    if chat_hits_only:
        for rank, (score, tid, eid) in enumerate(chat_hits_only, 1):
            text = chat_store.read_text(tid)
            preview = _format_preview(text, 220)
            print(f"[TOP {rank}] src=chat score={score:.4f} text_seg={tid} emb_seg={eid}")
            print(f"  text_path={chat_dir}/segment_{tid}.is")
            print(f"  vec_path={chat_dir}/segment_{eid}.is")
            print(f"  preview: {preview}\n")
    else:
        print("(no chat hits)")


def cmd_chat(args) -> None:
    wiki_dir, chat_dir = _resolve_dirs(args)
    if args.verbose:
        print(f"[INFO] wiki_dir={wiki_dir} chat_dir={chat_dir}")

    wiki_store = ISStore(wiki_dir)
    chat_store = ISStore(chat_dir)
    client = get_client()

    if not wiki_store.list_pairs():
        print(f"\n[WARN] No Wikipedia knowledge base found in {wiki_dir}")
        print(f"  Chat will work but won't have context to answer from.")
        print(f"  To add knowledge: ct build-wiki --topics-file topics.txt --limit 3\n")

    print("Chat is ready, type 'exit' to leave.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break

        qvec = embed_text_1536(
            client,
            user,
            chunk_tokens=args.embed_chunk_tokens,
            chunk_overlap=args.embed_chunk_overlap,
            pool=args.embed_pool,
            verbose=args.verbose,
        )

        convo_blocks, convo_tids = _read_recent_chat_window(chat_store, window=8)
        convo_blocks = [f"{ln}" for ln in convo_blocks]

        k_use = args.topk
        stride = 1
        if getattr(args, "energy_aware", False):
            policy = EnergyManager()
            d = policy.decide_params(k=args.topk, max_power_budget=args.max_power_budget)
            k_use, stride = d["k"], d["stride"]
            if args.verbose:
                print(f"[ENERGY] adapted topk={k_use} stride={stride} quant={d['quant_level']}")

        hits = _call_combined_search(
            user,
            qvec,
            wiki_store,
            chat_store,
            top_k=k_use,
            alpha=args.alpha,
            stride=stride,
            min_vec=args.min_score,
            min_lex=args.min_lex,
            min_hybrid=args.min_hybrid,
            use_shards=True,
            verbose=args.verbose,
        )

        use_context, raw_blocks = _call_select_relevant_blocks(
            user,
            hits,
            wiki_store,
            chat_store,
            topk=k_use,
            max_blocks=args.max_context_blocks,
            max_preview_chars=1200,
            alpha=args.alpha,
            min_vec=args.min_score,
            min_lex=args.min_lex,
            min_hybrid=args.min_hybrid,
        )

        knowledge_blocks: List[str] = []
        for b in raw_blocks:
            tid = _extract_tid_from_block(b)
            if tid is not None and tid in convo_tids:
                continue
            knowledge_blocks.append(b)

        knowledge_blocks = _filter_knowledge_blocks(knowledge_blocks, prefer_assistant=True)
        knowledge_blocks = _dedupe_blocks(knowledge_blocks)

        sections: List[str] = []

        sys_guidance = (
            "You are a helpful assistant. Follow this strictly:\n"
            "- Respond to the latest user message naturally and directly.\n"
            "- You are given two sections:\n"
            "  (A) USER CHAT — the recent conversation; do not summarize it unless asked.\n"
            "  (B) KNOWLEDGE — optional retrieved context; use it only to add factual details.\n"
            "- Do NOT describe the sections themselves. Do NOT recap messages unless the user asks.\n"
            "- If KNOWLEDGE conflicts with USER CHAT instructions, follow the user.\n"
            "- If the question is conversational, answer conversationally; if it is factual, you may quote specific facts from KNOWLEDGE.\n"
        )

        if convo_blocks:
            convo_text = "\n".join(convo_blocks)
            sections.append("=== USER CHAT (recent turns) ===\n" + convo_text)

        if use_context and knowledge_blocks:
            wiki_ctx, chat_ctx = [], []
            for b in knowledge_blocks:
                src = _block_source(b, wiki_dir, chat_dir)
                if src == "WIKI":
                    wiki_ctx.append(b)
                else:
                    chat_ctx.append(b)
            labeled = []
            if wiki_ctx:
                labeled += _label_blocks(wiki_ctx, "WIKI CONTEXT")
            if chat_ctx:
                labeled += _label_blocks(chat_ctx, "CHAT MEMORY")
            if labeled:
                sections.append("=== KNOWLEDGE (retrieved context) ===\n" + "\n\n".join(labeled))

        wiki_hits_only = _search_store_only(wiki_store, qvec, top_k=max(1, min(k_use, 5)))
        chat_hits_only = _search_store_only(chat_store, qvec, top_k=max(1, min(k_use, 5)))

        def _render_unfused(store: ISStore, base_dir: str, hits_limited):
            blocks = []
            for (score, tid, eid) in hits_limited:
                txt = store.read_text(tid)
                preview = _format_preview(txt, 600)
                blocks.append(
                    f"(text_path={base_dir}/segment_{tid}.is | vec_path={base_dir}/segment_{eid}.is | vec={score:.4f})\n\n{preview}"
                )
            return blocks

        extra_sections = []
        if wiki_hits_only:
            extra_sections.append(
                "=== WIKI (unfused top-k) ===\n" + "\n\n".join(_render_unfused(wiki_store, wiki_dir, wiki_hits_only))
            )
        if chat_hits_only:
            extra_sections.append(
                "=== CHAT (unfused top-k) ===\n" + "\n\n".join(_render_unfused(chat_store, chat_dir, chat_hits_only))
            )
        if extra_sections:
            sections.append("\n".join(extra_sections))

        tail = (
            "\n=== INSTRUCTIONS ===\n"
            "Answer the user directly. Do not explain how you used the sections. "
            "Only include facts from KNOWLEDGE if they improve the answer.\n\n"
            f"User: {user}"
        )

        prompt = sys_guidance + "\n\n".join(sections) + tail

        if args.verbose:
            if use_context and knowledge_blocks:
                print("\n===== SELECTED CONTEXT (KNOWLEDGE) =====")
                print("\n\n".join(knowledge_blocks))
            else:
                print("\n===== NO KNOWLEDGE SELECTED (or filtered) =====")
            print("===== FULL PROMPT SENT =====")
            print(prompt)
            _save_debug(hits, knowledge_blocks, prompt)

        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=350,
            )
            answer = r.choices[0].message.content.strip()
            print(f"AI: {answer}\n")
        except Exception as e:
            answer = f"(error: {e})"
            print(f"AI: {answer}\n")

        store_chat_turn(chat_store, "user", user, client=client, verbose=args.verbose)
        store_chat_turn(chat_store, "assistant", answer, client=client, verbose=args.verbose)


def cmd_stat(args) -> None:
    wiki_dir, chat_dir = _resolve_dirs(args)
    wiki_store = ISStore(wiki_dir)
    chat_store = ISStore(chat_dir)
    ws = wiki_store.stat()
    cs = chat_store.stat()

    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    total_tokens = 0

    for tid, _ in wiki_store.list_pairs():
        text = wiki_store.read_text(tid)
        total_tokens += len(enc.encode(text))
    for tid, _ in chat_store.list_pairs():
        text = chat_store.read_text(tid)
        total_tokens += len(enc.encode(text))

    print(
        f"[STAT] WIKI dir={wiki_dir} texts={ws['text_segments']} "
        f"vecs={ws['vector_segments']} pairs={ws['pairs']} next_id={ws['next_id']}"
    )
    print(
        f"[STAT] CHAT dir={chat_dir} texts={cs['text_segments']} "
        f"vecs={cs['vector_segments']} pairs={cs['pairs']} next_id={cs['next_id']}"
    )
    print(f"[STAT] TOKENS total={total_tokens:,} (~{total_tokens/1000:.1f}k)")


def cmd_bench(args) -> None:
    wiki_dir, chat_dir = _resolve_dirs(args)

    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [ln.strip() for ln in f if ln.strip()]
    else:
        queries = [
            "photosynthesis",
            "quantum computing applications",
            "internet protocols",
            "renaissance art",
        ]

    if args.verbose:
        print(f"[INFO] bench wiki_dir={wiki_dir} chat_dir={chat_dir}")
        print(
            f"[INFO] queries={len(queries)} repeats={args.repeats} topk={args.topk} energy_aware={args.energy_aware}"
        )

    res = run_benchmark(
        wiki_dir=wiki_dir,
        chat_dir=chat_dir,
        queries=queries[: args.max_queries] if args.max_queries else queries,
        repeats=args.repeats,
        topk=args.topk,
        energy_aware=args.energy_aware,
        max_power_budget=args.max_power_budget,
    )

    if res.energy_backend == "unavailable" and args.assume_power_watts:
        res.energy_backend = "assumed_power"
        res.energy_joules_pkg = args.assume_power_watts * res.elapsed_seconds

    print("\n=== Benchmark Report ===")
    print(f"Stores: wiki={res.wiki_dir} chat={res.chat_dir}")
    print(f"Disk: total {res.size_total_mb:.2f} MB (wiki {res.size_wiki_mb:.2f}, chat {res.size_chat_mb:.2f})")
    print(f"Segments: text={res.total_text_segments} vectors={res.total_vector_segments} pairs={res.total_pairs}")
    print(f"Tokens: total={res.total_tokens:,}  mean/chunk={res.tokens_per_chunk_mean:.1f}")
    print(f"Queries: {res.queries}  Repeats: {res.repeats}  topk: {res.topk}  energy_aware={res.energy_aware}")
    print(f"Latency (ms): p50={res.wall_ms_p50:.2f}  p95={res.wall_ms_p95:.2f}  mean={res.wall_ms_mean:.2f}")
    print(f"Throughput: mean QPS={res.qps_mean:.2f}")
    print(f"Memory: RSS max={res.rss_mb_max:.2f} MB" if res.rss_mb_max is not None else "Memory: RSS max=unavailable")
    print(f"        PSS max={res.pss_mb_max:.2f} MB" if res.pss_mb_max is not None else "        PSS max=unavailable")
    print(
        f"Energy: backend={res.energy_backend}"
        f"{(' pkg=%.3f J' % res.energy_joules_pkg) if res.energy_joules_pkg is not None else ''}"
        f"{(' dram=%.3f J' % res.energy_joules_dram) if res.energy_joules_dram is not None else ''}"
    )

    save_benchmark(res, args.out_json, args.out_csv)
    if args.out_md:
        md = render_benchmark_markdown(res)
        save_benchmark_markdown(md, args.out_md)
    if args.out_json or args.out_csv or args.out_md:
        saved_parts = []
        if args.out_json:
            saved_parts.append(f"JSON={args.out_json}")
        if args.out_csv:
            saved_parts.append(f"CSV={args.out_csv}")
        if args.out_md:
            saved_parts.append(f"MD={args.out_md}")
        print("\nSaved: " + " ".join(saved_parts))


def cmd_reset_chat(args) -> None:
    import shutil
    _, chat_dir = _resolve_dirs(args)
    if os.path.exists(chat_dir):
        shutil.rmtree(chat_dir)
    os.makedirs(chat_dir, exist_ok=True)
    print(f"[RESET] cleared {chat_dir}")


def cmd_ingest_any(args) -> None:
    out_dir = args.out_dir or WIKI_IS_DIR
    store = ISStore(out_dir)
    total = 0

    paths = []
    if os.path.isdir(args.path):
        for root, _, files in os.walk(args.path):
            for f in files:
                paths.append(os.path.join(root, f))
    else:
        paths.append(args.path)

    for pth in sorted(paths):
        if args.verbose:
            print(f"[INGEST-ANY] {pth}")
        try:
            tid, vid = ingest_any(store, pth, quantize_vec=args.quantize, max_bytes=args.max_bytes)
            total += 1
            if args.verbose:
                print(f"  -> text={tid} vec={vid}")
        except Exception as e:
            print(f"[SKIP] {pth} ({e})")

    stats = store.stat()
    print(
        f"[DONE] dir={store.dir_path} files_ingested={total} pairs={stats['pairs']} "
        f"texts={stats['text_segments']} vecs={stats['vector_segments']}"
    )


# ----------------------------------------------------------------------
# Main (rich help + examples)
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        prog="ct",
        description=(
            "ContextTape CLI – build simple on-disk text/vector segment stores and run hybrid RAG.\n"
            "Use the subcommands below. Each subcommand has examples and templates."
        ),
        epilog=HELP_EPILOG,
        formatter_class=RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = p.add_subparsers(dest="cmd")

    # Build wiki showcase (alias kept for backwards compat)
    s_build = sub.add_parser(
        "build-wiki",
        aliases=["build-wikipedia"],
        help="Ingest Wikipedia pages by title",
        description="""\
Fetch by titles (one per line) and write text+embedding pairs.

Examples
--------
  ct build-wiki --topics-file scripts/topics.example.txt --out-dir wiki_store --limit 50 --verbose

Templates
---------
  ct build-wiki --topics-file <path> [--out-dir DIR] [--limit N] [--min-chars K] [--verbose]
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    s_build.add_argument("--topics-file", required=True, help="Text file with one Wikipedia title per line")
    s_build.add_argument("--out-dir", default=WIKI_IS_DIR, help="Target store directory (default: wiki store)")
    s_build.add_argument("--limit", type=int, default=None, help="Limit number of titles")
    s_build.add_argument("--min-chars", type=int, default=0, help="Skip pages with fewer characters than this")
    s_build.add_argument("--skip-fences", action="store_true", default=True, help="Skip lines starting with ```")
    # Embedding knobs
    s_build.add_argument("--embed-chunk-tokens", type=int, default=int(os.getenv("EMBED_CHUNK_TOKENS", "800")),
                         help="Token window per chunk when text exceeds model limit (default 800).")
    s_build.add_argument("--embed-chunk-overlap", type=int, default=int(os.getenv("EMBED_CHUNK_OVERLAP", "200")),
                         help="Token overlap between chunks (default 200).")
    s_build.add_argument("--embed-pool", choices=["mean", "max", "weighted"], default=os.getenv("EMBED_POOL", "mean"),
                         help="How to pool chunk vectors into one vector (default: mean).")
    s_build.add_argument("--verbose", action="store_true")
    s_build.set_defaults(func=cmd_build_wiki)

    # Generic folder ingestion
    s_ing = sub.add_parser(
        "ingest-path",
        help="Ingest all files under a path",
        description=f"""\
Walk a file/folder and ingest text-like content (and PDFs if supported).

Examples
--------
  # Ingest markdown + txt + pdf with page cap
  ct ingest-path ./docs --out-dir wiki_store --exts md txt pdf --max-pdf-pages 20 --verbose

  # Read at most 1MB per file
  ct ingest-path ./logs --out-dir wiki_store --max-bytes 1048576

Templates
---------
  ct ingest-path <path> [--out-dir DIR] [--exts E1 E2 ...] [--max-bytes N] [--max-pdf-pages N]
                 [--follow-symlinks] [--embed-chunk-tokens K] [--embed-chunk-overlap K]
                 [--embed-pool mean|max|weighted] [--verbose]
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    s_ing.add_argument("path", help="File or folder to ingest")
    s_ing.add_argument("--out-dir", default=WIKI_IS_DIR, help="Target store directory (default: wiki store)")
    s_ing.add_argument("--exts", nargs="*", help="Extensions to include, e.g., --exts md txt pdf")
    s_ing.add_argument("--max-bytes", type=int, default=None, help="Read cap per text file")
    s_ing.add_argument("--max-pdf-pages", type=int, default=20, help="Max pages to read per PDF")
    s_ing.add_argument("--follow-symlinks", action="store_true")
    # Embedding knobs
    s_ing.add_argument("--embed-chunk-tokens", type=int, default=int(os.getenv("EMBED_CHUNK_TOKENS", "800")))
    s_ing.add_argument("--embed-chunk-overlap", type=int, default=int(os.getenv("EMBED_CHUNK_OVERLAP", "200")))
    s_ing.add_argument("--embed-pool", choices=["mean", "max", "weighted"], default=os.getenv("EMBED_POOL", "mean"))
    s_ing.add_argument("--verbose", action="store_true")
    s_ing.set_defaults(func=cmd_ingest_path)

    # Search
    s_search = sub.add_parser(
        "search",
        help="Search across wiki and chat memory (env: WIKI_IS_DIR, CHAT_IS_DIR)",
        description=f"""\
Show per-store results (WIKI and CHAT) using vector similarity (no fusion printing).

Examples
--------
  ct search "quantum entanglement applications" --topk 8 --verbose
  ct search "fire propagation modeling" --wiki-dir data/wiki --chat-dir data/chat

Templates
---------
  ct search "<query text...>" [--topk K] [--wiki-topk K] [--chat-topk K] [--wiki-dir DIR] [--chat-dir DIR]
            [--embed-chunk-tokens K] [--embed-chunk-overlap K] [--embed-pool mean|max|weighted]
            [--energy-aware] [--max-power-budget W] [--verbose]
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    s_search.add_argument("query", nargs="+")
    s_search.add_argument("--topk", type=int, default=TOP_K_DEFAULT)
    s_search.add_argument("--wiki-topk", type=int, default=None, help="Override top-k for wiki-only section")
    s_search.add_argument("--chat-topk", type=int, default=None, help="Override top-k for chat-only section")
    s_search.add_argument("--wiki-dir", help="Directory for the wiki store (overrides WIKI_IS_DIR)")
    s_search.add_argument("--chat-dir", help="Directory for the chat store (overrides CHAT_IS_DIR)")
    # Embedding knobs (mostly irrelevant for short queries but harmless)
    s_search.add_argument("--embed-chunk-tokens", type=int, default=int(os.getenv("EMBED_CHUNK_TOKENS", "800")))
    s_search.add_argument("--embed-chunk-overlap", type=int, default=int(os.getenv("EMBED_CHUNK_OVERLAP", "200")))
    s_search.add_argument("--embed-pool", choices=["mean", "max", "weighted"], default=os.getenv("EMBED_POOL", "mean"))
    s_search.add_argument("--energy-aware", action="store_true", help="Adapt k/stride to save energy")
    s_search.add_argument("--max-power-budget", type=float, default=None, help="If set, throttle when above this W")
    s_search.add_argument("--verbose", action="store_true")
    s_search.set_defaults(func=cmd_search)

    # Chat
    s_chat = sub.add_parser(
        "chat",
        help="Interactive chat using retrieved context (env: WIKI_IS_DIR, CHAT_IS_DIR)",
        description="""\
RAG chat that retrieves/contextualizes from WIKI + CHAT stores, then calls a chat model.

Key knobs
---------
  --alpha         Hybrid blend: hybrid = alpha*vector + (1-alpha)*lexical
                  1.0 = pure vector, 0.0 = pure lexical.
  --min-score     Minimum vector similarity to keep a block (start 0.30–0.35).
  --min-lex       Minimum lexical overlap to keep a block (start 0.10–0.15).
  --min-hybrid    Minimum final hybrid score to keep a block (start 0.25–0.30).
  --max-context-blocks   Hard cap on blocks passed into the prompt (start 5).

Recipes
-------
  # Keyword-focused:
  ct chat --alpha 0.40 --min-lex 0.18 --min-hybrid 0.32 --topk 10 --max-context-blocks 6 --verbose

  # Semantic/looser phrasing:
  ct chat --alpha 0.70 --min-score 0.30 --min-lex 0.10 --min-hybrid 0.26 --topk 12 --max-context-blocks 6 --verbose

  # Battery-saver:
  ct chat --topk 6 --max-context-blocks 4 --energy-aware --max-power-budget 12 --verbose

Templates
---------
  ct chat [--wiki-dir DIR] [--chat-dir DIR] [--topk K] [--alpha A]
          [--min-score X] [--min-lex Y] [--min-hybrid Z] [--max-context-blocks N]
          [--embed-chunk-tokens K] [--embed-chunk-overlap K] [--embed-pool mean|max|weighted]
          [--energy-aware] [--max-power-budget W] [--verbose]
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    s_chat.add_argument("--topk", type=int, default=TOP_K_DEFAULT)
    s_chat.add_argument("--wiki-dir", help="Directory for the wiki store (overrides WIKI_IS_DIR)")
    s_chat.add_argument("--chat-dir", help="Directory for the chat store (overrides CHAT_IS_DIR)")
    s_chat.add_argument("--min-score", type=float, default=float(os.getenv("MIN_SCORE", "0.32")),
                        help="Minimum vector similarity to include a block")
    s_chat.add_argument("--min-lex", type=float, default=float(os.getenv("MIN_LEX", "0.12")),
                        help="Minimum lexical overlap to include a block")
    s_chat.add_argument("--min-hybrid", type=float, default=float(os.getenv("MIN_HYBRID", "0.28")),
                        help="Minimum α·vec+(1−α)·lex to include a block")
    s_chat.add_argument("--alpha", type=float, default=float(os.getenv("HYBRID_ALPHA", "0.7")),
                        help="Hybrid α weight for vector vs lexical")
    s_chat.add_argument("--max-context-blocks", type=int, default=None,
                        help="Cap number of retrieved blocks (<= topk)")
    # Embedding knobs (applied to query embedding; context embeddings are precomputed)
    s_chat.add_argument("--embed-chunk-tokens", type=int, default=int(os.getenv("EMBED_CHUNK_TOKENS", "800")))
    s_chat.add_argument("--embed-chunk-overlap", type=int, default=int(os.getenv("EMBED_CHUNK_OVERLAP", "200")))
    s_chat.add_argument("--embed-pool", choices=["mean", "max", "weighted"], default=os.getenv("EMBED_POOL", "mean"))
    s_chat.add_argument("--energy-aware", action="store_true", help="Adapt k/stride in chat to save energy")
    s_chat.add_argument("--max-power-budget", type=float, default=None, help="If set, throttle when above this W")
    s_chat.add_argument("--verbose", action="store_true")
    s_chat.set_defaults(func=cmd_chat)

    # Stats
    s_stat = sub.add_parser(
        "stat",
        help="Show store stats (env: WIKI_IS_DIR, CHAT_IS_DIR)",
        description="""\
Print counts, next_id, and exact token totals.

Examples
--------
  ct stat --wiki-dir data/wiki --chat-dir data/chat
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    s_stat.add_argument("--wiki-dir", help="Directory for the wiki store (overrides WIKI_IS_DIR)")
    s_stat.add_argument("--chat-dir", help="Directory for the chat store (overrides CHAT_IS_DIR)")
    s_stat.set_defaults(func=cmd_stat)

    # Benchmark
    s_bench = sub.add_parser(
        "bench",
        help="Benchmark: tokens, size, speed, RAM, energy",
        description="""\
Run a small performance/size/energy benchmark and save reports.

Examples
--------
  ct bench --wiki-dir data/wiki --chat-dir data/chat \
           --queries-file ./queries.txt --max-queries 50 --repeats 5 \
           --topk 8 --energy-aware --max-power-budget 15 \
           --assume-power-watts 15 --out-json bench.json --out-csv bench.csv --out-md bench.md --verbose

Report fields (high-value)
--------------------------
  Latency (p50/p95/mean), QPS
  Memory (RSS/PSS) when available
  Energy backend + Joules (pkg/dram or assumed)
  Store sizes (MB), tokens, pairs

Templates
---------
  ct bench [--wiki-dir DIR] [--chat-dir DIR] [--queries-file FILE] [--max-queries N]
           [--repeats N] [--topk K] [--energy-aware] [--max-power-budget W]
           [--assume-power-watts W] [--out-json PATH] [--out-csv PATH] [--out-md PATH] [--verbose]
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    s_bench.add_argument("--wiki-dir", help="Directory for the wiki store (overrides WIKI_IS_DIR)")
    s_bench.add_argument("--chat-dir", help="Directory for the chat store (overrides CHAT_IS_DIR)")
    s_bench.add_argument("--queries-file", help="Path to a newline-separated list of queries")
    s_bench.add_argument("--max-queries", type=int, default=None, help="Cap number of queries used")
    s_bench.add_argument("--repeats", type=int, default=3, help="Number of passes over the query set")
    s_bench.add_argument("--topk", type=int, default=TOP_K_DEFAULT)
    s_bench.add_argument("--energy-aware", action="store_true", help="Adapt k/stride to save energy")
    s_bench.add_argument("--max-power-budget", type=float, default=None, help="If set, throttle when above this W")
    s_bench.add_argument("--assume-power-watts", type=float, default=None,
                         help="If hardware energy meters are unavailable, estimate energy as Watts * elapsed_time")
    s_bench.add_argument("--out-json", help="Save benchmark results to JSON")
    s_bench.add_argument("--out-csv", help="Save benchmark results to CSV")
    s_bench.add_argument("--out-md", help="Save a Markdown patent-style report")
    s_bench.add_argument("--verbose", action="store_true")
    s_bench.set_defaults(func=cmd_bench)

    # Reset chat store
    s_reset = sub.add_parser(
        "reset-chat",
        help="Clear chat store (env: CHAT_IS_DIR)",
        description="""\
Delete and recreate the chat store directory only.

Examples
--------
  ct reset-chat --chat-dir data/chat
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    s_reset.add_argument("--chat-dir", help="Directory for the chat store (overrides CHAT_IS_DIR)")
    s_reset.set_defaults(func=cmd_reset_chat)

    # Ingest-any (mixed modalities)
    s_ing_any = sub.add_parser(
        "ingest-any",
        help="Ingest arbitrary files (text/image/audio/pdf/video/other blobs)",
        description="""\
Best-effort multimodal ingestion. Text is embedded as text; images/audio are embedded as media;
PDFs optionally extract text (if supported); video defaults to filename-based summary.
Other types fall back to a blob manifest with a filename summary.

Examples
--------
  # Seed & ingest sample corpus (text, md, json, csv, png, jpg, wav, dummy mp4)
  python scripts/seed_multimodal_corpus.py --out sample_corpus
  ct ingest-any sample_corpus --out-dir wiki_store --quantize --verbose

  # Ingest a single image or audio
  ct ingest-any ./image.png --out-dir wiki_store
  ct ingest-any ./tone.wav  --out-dir wiki_store

Templates
---------
  ct ingest-any <path> [--out-dir DIR] [--quantize] [--max-bytes N] [--verbose]
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    s_ing_any.add_argument("path", help="File or folder to ingest")
    s_ing_any.add_argument("--out-dir", default=WIKI_IS_DIR, help="Target store directory (default: wiki store)")
    s_ing_any.add_argument("--quantize", action="store_true", help="Store vectors as int8 to save space")
    s_ing_any.add_argument("--max-bytes", type=int, default=None, help="Read cap per binary/text file")
    s_ing_any.add_argument("--verbose", action="store_true")
    s_ing_any.set_defaults(func=cmd_ingest_any)

    args = p.parse_args()
    if not getattr(args, "cmd", None):
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
