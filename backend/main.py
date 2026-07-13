import os

import io
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from matcher import analyze, generate_chat_reply

from fastapi.responses import StreamingResponse
from pdf_export import build_resume_pdf
 

app = FastAPI(title="JobFit AI", version="1.0.0")

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")

    if not job_description or len(job_description.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Job description looks too short — paste the full listing.",
        )

    resume_bytes = await resume.read()

    try:
        result = analyze(resume_bytes, job_description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return result



from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    resume_text: str
    job_description: str
    messages: list[ChatMessage]


@app.post("/chat")
async def chat(payload: ChatRequest):
    try:
        reply = generate_chat_reply(
            payload.resume_text,
            payload.job_description,
            [m.model_dump() for m in payload.messages],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    return {"reply": reply}

class PdfExportRequest(BaseModel):
    resume_text: str


@app.post("/export-resume-pdf")
async def export_resume_pdf(payload: PdfExportRequest):
    if not payload.resume_text.strip():
        raise HTTPException(status_code=400, detail="No resume text to export.")

    try:
        pdf_bytes = build_resume_pdf(payload.resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=tailored_resume.pdf"},
    )



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
