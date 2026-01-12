# Changelog

All notable changes to ContextTape will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

---

## [0.5.0] - 2026-01-08

### Added
- **Int8 quantization**: Store vectors with `--quantize` for 4× smaller storage
- **Multi-store fusion**: `MultiStore` class for searching across multiple directories
- **Coarse prefiltering**: Optional centroid-based filtering for faster scans
- **Energy-aware mode**: Auto-tune top-k/stride under power constraints
- **Multimodal support**: Ingest images, audio, PDFs with manifest segments
- **JSON segments**: `DT_JSON` type for structured metadata storage
- **Media + manifest pattern**: `append_media_with_manifest()` for rich media
- **Benchmark CLI**: `ct bench` with latency/throughput/memory/energy reports
- **Hybrid reranking**: Vector similarity + lexical overlap scoring

### Changed
- Improved CLI with `--verbose` debug output (hits.json, context.md, prompt.txt)
- Better error messages and help text
- Reorganized optional dependencies (pdf, energy, all)

### Fixed
- Memory leak in repeated searches
- Header timestamp encoding (now uses reserved 8 bytes)

---

## [0.4.0] - 2025-12-15

### Added
- **Chat memory store**: Separate store for conversation history
- **Wiki ingester**: `ct build-wiki` to bootstrap from Wikipedia
- **Path ingester**: `ct ingest-path` for local documents
- **PDF support**: Optional `pypdf` integration
- **Search command**: `ct search` with cosine similarity
- **Chat command**: `ct chat` with retrieval-augmented generation

### Changed
- Header format now 32 bytes (added reserved field for timestamp)
- Improved chunking with configurable overlap

---

## [0.3.0] - 2025-11-20

### Added
- **ISStore class**: Core segment store implementation
- **ISHeader**: 32-byte binary header with linking
- **Data types**: DT_TEXT, DT_VEC_F32, DT_VEC_I8
- **Basic search**: `search_by_vector()` with top-k
- **Embedding utilities**: OpenAI text-embedding-3-small integration

### Changed
- Switched from JSON metadata to binary headers

---

## [0.2.0] - 2025-10-15

### Added
- Initial CLI prototype
- Basic file ingestion
- Simple cosine search

---

## [0.1.0] - 2025-09-01

### Added
- Project inception
- Proof of concept for file-based RAG storage

---

[Unreleased]: https://github.com/NuTerraLabs/contexttape/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/NuTerraLabs/contexttape/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/NuTerraLabs/contexttape/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NuTerraLabs/contexttape/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NuTerraLabs/contexttape/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NuTerraLabs/contexttape/releases/tag/v0.1.0
