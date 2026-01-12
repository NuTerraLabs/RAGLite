ContextTape Demo Corpus
=======================
This folder contains a deliberately diverse set of files so your retriever has meaningful
context to return. Try queries like:
- photosynthesis light-dependent reactions ATP NADPH Calvin cycle
- HTTP vs HTTPS TLS handshake status codes idempotent methods
- Renaissance art Medici patronage perspective Brunelleschi
- Kamala Harris Senate priorities criminal justice immigration healthcare

Files:
- notes.md                 (science: photosynthesis, with equations)
- http_https.md            (networking: protocol explainer)
- renaissance_art.md       (humanities: concise overview)
- kamala_harris.txt        (bio/context prior to VP term end in Jan 2025)
- kb_index.json            (structured summaries for each topic)
- glossary.csv             (keyword → short blurb mapping for quick hits)
- pixel.png / pixel.jpg    (1x1 images; used to test image ingest path)
- tone.wav                 (1s audio tone; tests audio ingest)
- clip.mp4                 (dummy blob; tests generic/video path)
