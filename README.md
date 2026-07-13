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

## Running locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then add your OPENAI_API_KEY
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`. Check `http://localhost:8000/health`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env       # defaults to localhost:8000, fine for local dev
npm run dev
```

Frontend runs at `http://localhost:5173`.

## Deploying (free tier, ~30 min)

1. **Backend → Render**
   - New Web Service → connect this repo, root directory `backend`
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Add environment variable `OPENAI_API_KEY`
   - Add `ALLOWED_ORIGINS` once you have the Vercel URL

2. **Frontend → Vercel**
   - Import repo, root directory `frontend`
   - Framework preset: Vite
   - Add environment variable `VITE_API_URL` = your Render backend URL

3. Update `ALLOWED_ORIGINS` on Render with the final Vercel URL and redeploy.

## Known limitations (good to mention proactively in interviews)

- Resume PDFs with heavy graphic/columnar layouts can extract text out of
  order — a production version would need layout-aware parsing.
- Embedding-based similarity rewards keyword/phrase overlap; it can miss
  cases where a candidate has equivalent experience described in very
  different language.
- Currently single-request, no persistence — a natural next step is storing
  past analyses per user (Postgres/Supabase) to show improvement over time.

## Possible extensions

- Save analysis history per user (adds a DB + auth story)
- Support `.docx` resumes
- Batch mode: one resume against multiple JDs to find the best-fit role
