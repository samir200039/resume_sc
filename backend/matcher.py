# """
# Core logic for JobFit AI.

# Pipeline:
# 1. Extract raw text from an uploaded resume PDF.
# 2. Split the resume into bullet-level chunks and the job description into
#    requirement-level chunks.
# 3. Embed every chunk with Gemini's embedding model.
# 4. Score the match with cosine similarity between resume chunks and JD
#    requirement chunks (this is the "real ML", not an LLM guess).
# 5. Ask Gemini's chat model for a structured gap analysis + rewritten
#    bullets, grounded in the similarity results from step 4.
# """

# import io
# import json
# import os
# import re

# import google.generativeai as genai
# import numpy as np
# import pdfplumber

# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# EMBEDDING_MODEL = "models/gemini-embedding-001"
# CHAT_MODEL = "gemini-2.0-flash"


# def extract_text_from_pdf(file_bytes: bytes) -> str:
#     text_parts = []
#     with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text() or ""
#             text_parts.append(page_text)
#     return "\n".join(text_parts).strip()


# def split_into_chunks(text: str, min_len: int = 15) -> list[str]:
#     lines = re.split(r"[\n\u2022\r]+", text)
#     chunks = [line.strip(" -\t") for line in lines]
#     return [c for c in chunks if len(c) >= min_len]


# def get_embeddings(chunks: list[str], task_type: str = "retrieval_document") -> np.ndarray:
#     if not chunks:
#         return np.zeros((0, 768))

#     vectors = []
#     for chunk in chunks:
#         result = genai.embed_content(
#             model=EMBEDDING_MODEL,
#             content=chunk,
#             task_type=task_type,
#         )
#         vectors.append(result["embedding"])
#     return np.array(vectors)


# def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
#     b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
#     return a_norm @ b_norm.T


# def compute_match_score(resume_chunks: list[str], jd_chunks: list[str]) -> dict:
#     resume_vecs = get_embeddings(resume_chunks, task_type="retrieval_document")
#     jd_vecs = get_embeddings(jd_chunks, task_type="retrieval_query")

#     if resume_vecs.shape[0] == 0 or jd_vecs.shape[0] == 0:
#         return {"overall_score": 0, "requirement_scores": []}

#     sim_matrix = cosine_similarity_matrix(jd_vecs, resume_vecs)
#     best_match_idx = sim_matrix.argmax(axis=1)
#     best_match_scores = sim_matrix.max(axis=1)

#     requirement_scores = []
#     for i, jd_line in enumerate(jd_chunks):
#         requirement_scores.append(
#             {
#                 "requirement": jd_line,
#                 "best_resume_match": resume_chunks[best_match_idx[i]],
#                 "similarity": round(float(best_match_scores[i]) * 100, 1),
#             }
#         )

#     overall_score = round(float(best_match_scores.mean()) * 100, 1)
#     return {"overall_score": overall_score, "requirement_scores": requirement_scores}


# def generate_gap_analysis(resume_text: str, jd_text: str, match_result: dict) -> dict:
#     weak_requirements = [
#         r["requirement"]
#         for r in sorted(match_result["requirement_scores"], key=lambda r: r["similarity"])[:5]
#     ]

#     prompt = f"""You are a resume reviewer. Given the resume and job description below,
# and the list of the weakest-matching job requirements, do three things:

# 1. List 3-5 concrete skills/keywords present in the job description but missing or weak in the resume.
# 2. List 3-5 strengths already well covered in the resume for this job.
# 3. Rewrite 3 resume bullet points (pick real bullets from the resume) to better target this job description.
#    Keep them truthful to the original experience, just sharper and more keyword-aligned.

# Weakest-matching requirements:
# {json.dumps(weak_requirements, indent=2)}

# Resume:
# {resume_text[:4000]}

# Job Description:
# {jd_text[:2000]}

# Respond with a JSON object in exactly this shape:
# {{
#   "missing_skills": ["..."],
#   "strengths": ["..."],
#   "rewritten_bullets": [
#     {{"original": "...", "rewritten": "..."}}
#   ]
# }}
# """

#     model = genai.GenerativeModel(CHAT_MODEL)
#     response = model.generate_content(
#         prompt,
#         generation_config={
#             "temperature": 0.4,
#             "response_mime_type": "application/json",
#         },
#     )

#     try:
#         return json.loads(response.text)
#     except (json.JSONDecodeError, ValueError):
#         return {
#             "missing_skills": [],
#             "strengths": [],
#             "rewritten_bullets": [],
#             "parse_error": "Model did not return valid JSON",
#         }


# def analyze(resume_pdf_bytes: bytes, jd_text: str) -> dict:
#     resume_text = extract_text_from_pdf(resume_pdf_bytes)
#     if not resume_text:
#         raise ValueError("Could not extract any text from the uploaded PDF.")

#     resume_chunks = split_into_chunks(resume_text)
#     jd_chunks = split_into_chunks(jd_text)

#     match_result = compute_match_score(resume_chunks, jd_chunks)
#     gap_analysis = generate_gap_analysis(resume_text, jd_text, match_result)

#     return {
#         "overall_score": match_result["overall_score"],
#         "requirement_breakdown": match_result["requirement_scores"],
#         "missing_skills": gap_analysis.get("missing_skills", []),
#         "strengths": gap_analysis.get("strengths", []),
#         "rewritten_bullets": gap_analysis.get("rewritten_bullets", []),
#     }



"""
Core logic for JobFit AI.

Pipeline:
1. Extract raw text from an uploaded resume PDF.
2. Split the resume into bullet-level chunks and the job description into
   requirement-level chunks.
3. Embed every chunk locally with sentence-transformers — no API call,
   so this step can never be rate-limited and has zero external dependency.
4. Score the match with cosine similarity between resume chunks and JD
   requirement chunks (this is the "real ML", not an LLM guess).
5. Ask an open-source Llama model via Groq for a structured gap analysis +
   rewritten bullets, grounded in the similarity results from step 4.
"""

import io
import json
import os
import re

import numpy as np
import pdfplumber
import requests
from sentence_transformers import SentenceTransformer

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# Loaded once at startup, runs locally on CPU — no API calls for embeddings.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def split_into_chunks(text: str, min_len: int = 15) -> list[str]:
    lines = re.split(r"[\n\u2022\r]+", text)
    chunks = [line.strip(" -\t") for line in lines]
    return [c for c in chunks if len(c) >= min_len]


def get_embeddings(chunks: list[str]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 384))
    return embedding_model.encode(chunks, convert_to_numpy=True)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def compute_match_score(resume_chunks: list[str], jd_chunks: list[str]) -> dict:
    resume_vecs = get_embeddings(resume_chunks)
    jd_vecs = get_embeddings(jd_chunks)

    if resume_vecs.shape[0] == 0 or jd_vecs.shape[0] == 0:
        return {"overall_score": 0, "requirement_scores": []}

    sim_matrix = cosine_similarity_matrix(jd_vecs, resume_vecs)
    best_match_idx = sim_matrix.argmax(axis=1)
    best_match_scores = sim_matrix.max(axis=1)

    requirement_scores = []
    for i, jd_line in enumerate(jd_chunks):
        requirement_scores.append(
            {
                "requirement": jd_line,
                "best_resume_match": resume_chunks[best_match_idx[i]],
                "similarity": round(float(best_match_scores[i]) * 100, 1),
            }
        )

    overall_score = round(float(best_match_scores.mean()) * 100, 1)
    return {"overall_score": overall_score, "requirement_scores": requirement_scores}


def generate_gap_analysis(resume_text: str, jd_text: str, match_result: dict) -> dict:
    weak_requirements = [
        r["requirement"]
        for r in sorted(match_result["requirement_scores"], key=lambda r: r["similarity"])[:5]
    ]

    prompt = f"""You are a resume reviewer. Given the resume and job description below,
and the list of the weakest-matching job requirements, do three things:

1. List 3-5 concrete skills/keywords present in the job description but missing or weak in the resume.
2. List 3-5 strengths already well covered in the resume for this job.
3. Rewrite 3 resume bullet points (pick real bullets from the resume) to better target this job description.
   Keep them truthful to the original experience, just sharper and more keyword-aligned.

Weakest-matching requirements:
{json.dumps(weak_requirements, indent=2)}

Resume:
{resume_text[:4000]}

Job Description:
{jd_text[:2000]}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "missing_skills": ["..."],
  "strengths": ["..."],
  "rewritten_bullets": [
    {{"original": "...", "rewritten": "..."}}
  ]
}}
"""

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    response.raise_for_status()
    raw_text = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "missing_skills": [],
            "strengths": [],
            "rewritten_bullets": [],
            "parse_error": "Model did not return valid JSON",
        }

def analyze(resume_pdf_bytes: bytes, jd_text: str) -> dict:
    resume_text = extract_text_from_pdf(resume_pdf_bytes)
    if not resume_text:
        raise ValueError("Could not extract any text from the uploaded PDF.")

    resume_chunks = split_into_chunks(resume_text)
    jd_chunks = split_into_chunks(jd_text)

    match_result = compute_match_score(resume_chunks, jd_chunks)
    gap_analysis = generate_gap_analysis(resume_text, jd_text, match_result)

    return {
        "overall_score": match_result["overall_score"],
        "requirement_breakdown": match_result["requirement_scores"],
        "missing_skills": gap_analysis.get("missing_skills", []),
        "strengths": gap_analysis.get("strengths", []),
        "rewritten_bullets": gap_analysis.get("rewritten_bullets", []),
        "resume_text": resume_text,
    }


def generate_chat_reply(resume_text: str, jd_text: str, messages: list[dict]) -> str:
    """
    Free-form follow-up chat, grounded in the original resume + JD.
    `messages` is the running conversation: [{"role": "user"/"assistant", "content": "..."}]
    """
    system_prompt = f"""You are a resume-writing assistant helping a candidate tailor their
resume to a specific job. You have their original resume and the target job description below.
When asked to write or rewrite a resume, produce a complete, well-formatted, plain-text resume
(use clear section headers and bullet points with "-"). Keep every claim truthful to the
candidate's actual background in the original resume — sharpen language and reorder/emphasize
relevant experience, but never invent degrees, companies, job titles, or skills they don't have.
Be direct and practical in all replies.

Original resume:
{resume_text[:4000]}

Target job description:
{jd_text[:2000]}
"""

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": 0.5,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]