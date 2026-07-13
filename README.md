# JobFit AI

Upload a resume (PDF) and paste a job description. JobFit AI scores how well
they actually match using embedding-based semantic similarity, then reports
which requirements are weakly covered, which skills are missing, and rewrites
a few resume bullets to close the gap.

## Why it's built this way (for interviews / README credibility)

The match score is **not** an LLM guessing a number — that's a black box and
inconsistent between runs. Instead:

1. The resume and job description are each split into line-level chunks.
2. Every chunk is embedded with OpenAI's `text-embedding-3-small`.
3. For every job requirement, we find its best-matching resume chunk using
   **cosine similarity** and average the best-match scores. This rewards
   covering every requirement, not just being strong in one area.
4. Only *after* the quantitative score exists does an LLM call run — and its
   job is narrow: explain the gaps and rewrite bullets, grounded in the
   weakest-scoring requirements from step 3. It never invents the score.

This split (deterministic ML for scoring, LLM for language) is the core
design decision worth explaining to a recruiter.

## Architecture

```
┌─────────────┐      multipart/form-data       ┌──────────────────┐
│   React      │ ──────────────────────────────▶│   FastAPI         │
│   (Vercel)   │                                 │   (Render/Railway)│
│              │◀──────────────────────────────  │                    │
└─────────────┘         JSON result              └─────────┬──────────┘
                                                             │
                                            ┌────────────────┼────────────────┐
                                            ▼                                 ▼
                                   pdfplumber (parse)              OpenAI API
                                                                (embeddings + chat)
```

## Project structure

```
jobfit-ai/
├── backend/
│   ├── main.py           # FastAPI app, /analyze endpoint
│   ├── matcher.py         # parsing, embeddings, cosine similarity, LLM gap analysis
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # upload form + score readout UI
│   │   ├── main.jsx
│   │   └── index.css       # design tokens + styles
│   ├── index.html
│   ├── package.json
│   └── .env.example
└── README.md
```
