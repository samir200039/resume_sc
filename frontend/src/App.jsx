import { useState, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const handleFile = useCallback((file) => {
    if (file && file.type === "application/pdf") {
      setResumeFile(file);
      setError(null);
    } else {
      setError("Please upload a PDF resume.");
    }
  }, []);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!resumeFile) {
      setError("Upload your resume as a PDF first.");
      return;
    }
    if (jobDescription.trim().length < 20) {
      setError("Paste the full job description — that's what gets matched against.");
      return;
    }

    setLoading(true);
    setResult(null);
    setChatMessages([]);

    const formData = new FormData();
    formData.append("resume", resumeFile);
    formData.append("job_description", jobDescription);

    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || "Analysis failed. Try again.");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const sendChatMessage = async (text) => {
    if (!text.trim() || !result) return;

    const updated = [...chatMessages, { role: "user", content: text }];
    setChatMessages(updated);
    setChatInput("");
    setChatLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume_text: result.resume_text,
          job_description: jobDescription,
          messages: updated,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || "Chat failed.");
      }

      const data = await res.json();
      setChatMessages([...updated, { role: "assistant", content: data.reply }]);
    } catch (err) {
      setChatMessages([
        ...updated,
        { role: "assistant", content: `Error: ${err.message}` },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const downloadResumePdf = async (text) => {
    try {
      const res = await fetch(`${API_URL}/export-resume-pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ resume_text: text }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || "PDF export failed.");
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "tailored_resume.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="eyebrow">jobfit ai / match engine</div>
        <h1 className="title">Does your resume actually fit the role?</h1>
        <p className="subtitle">
          Upload your resume and paste a job description. JobFit scores the real
          semantic overlap with embeddings, then tells you exactly what's missing
          and how to fix it.
        </p>
      </header>

      <form className="card" onSubmit={handleSubmit}>
        <div className="field">
          <label htmlFor="resume-upload">Resume (PDF)</label>
          <div
            id="resume-upload"
            className={`dropzone ${isDragging ? "active" : ""}`}
            tabIndex={0}
            role="button"
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById("file-input").click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                document.getElementById("file-input").click();
              }
            }}
          >
            {resumeFile ? (
              <>
                Ready to analyze
                <div className="filename">{resumeFile.name}</div>
              </>
            ) : (
              "Drag & drop your resume, or click to browse"
            )}
            <input
              id="file-input"
              type="file"
              accept="application/pdf"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>
        </div>

        <div className="field">
          <label htmlFor="jd">Job description</label>
          <textarea
            id="jd"
            placeholder="Paste the full job listing here — responsibilities and requirements both matter."
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
          />
        </div>

        {error && <div className="error-banner">{error}</div>}

        <button className="submit-btn" type="submit" disabled={loading}>
          {loading ? "Analyzing…" : "Run match analysis"}
        </button>
      </form>

      {loading && <div className="loading">Embedding resume and job description…</div>}

      {result && (
        <div className="readout">
          <div className="readout-scan">
            <div className="score-value">{result.overall_score}%</div>
            <div className="score-label">Semantic match score</div>
          </div>

          <div className="readout-body">
            {result.missing_skills?.length > 0 && (
              <>
                <h3 className="section-title missing">Missing from your resume</h3>
                <div className="tag-list">
                  {result.missing_skills.map((skill, i) => (
                    <span className="tag missing" key={i}>
                      {skill}
                    </span>
                  ))}
                </div>
              </>
            )}

            {result.strengths?.length > 0 && (
              <>
                <h3 className="section-title strength">Already covered well</h3>
                <div className="tag-list">
                  {result.strengths.map((skill, i) => (
                    <span className="tag strength" key={i}>
                      {skill}
                    </span>
                  ))}
                </div>
              </>
            )}

            {result.rewritten_bullets?.length > 0 && (
              <>
                <h3 className="section-title bullets">Suggested rewrites</h3>
                {result.rewritten_bullets.map((b, i) => (
                  <div className="bullet-pair" key={i}>
                    <div className="original">{b.original}</div>
                    <div className="rewritten">{b.rewritten}</div>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      )}

      {result && (
        <div className="chat-panel">
          <h3 className="section-title bullets">Ask JobFit</h3>

          {chatMessages.length === 0 && (
            <button
              type="button"
              className="quick-action-btn"
              onClick={() =>
                sendChatMessage("Write a full resume tailored to this job description.")
              }
            >
              Generate a resume tailored to this JD
            </button>
          )}

          <div className="chat-messages">
            {chatMessages.map((m, i) => (
              <div key={i} className={`chat-bubble ${m.role}`}>
                <div className="chat-role">{m.role === "user" ? "You" : "JobFit"}</div>
                <pre className="chat-content">{m.content}</pre>
                {m.role === "assistant" && (
                  <button
                    type="button"
                    className="download-btn"
                    onClick={() => downloadResumePdf(m.content)}
                  >
                    ⬇ Download as PDF
                  </button>
                )}
              </div>
            ))}
            {chatLoading && <div className="loading">Thinking…</div>}
          </div>

          <form
            className="chat-input-row"
            onSubmit={(e) => {
              e.preventDefault();
              sendChatMessage(chatInput);
            }}
          >
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="e.g. Make bullet 2 punchier, or shorten this to one page"
            />
            <button type="submit" disabled={chatLoading || !chatInput.trim()}>
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
}