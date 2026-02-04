import json
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def clamp_text(s: str, max_chars: int) -> str:
    s = s.strip()
    return s[:max_chars] if len(s) > max_chars else s

def empty_payload() -> Dict[str, Any]:
    return {
        "summary": {"breathing": [], "feeding": [], "growth": [], "events": []},
        "questions": {"breathing": [], "feeding": [], "growth": [], "events": [], "discharge": []},
    }

def enforce_shape(obj: Any) -> Dict[str, Any]:
    base = empty_payload()
    if not isinstance(obj, dict):
        return base

    for sec in ["breathing", "feeding", "growth", "events"]:
        arr = obj.get("summary", {}).get(sec)
        if isinstance(arr, list):
            base["summary"][sec] = [str(x) for x in arr if isinstance(x, str)]

    for sec in ["breathing", "feeding", "growth", "events", "discharge"]:
        arr = obj.get("questions", {}).get(sec)
        if isinstance(arr, list):
            base["questions"][sec] = [str(x) for x in arr if isinstance(x, str)]

    # Cap total questions at 12
    total = 0
    for sec in ["breathing", "feeding", "growth", "events", "discharge"]:
        trimmed: List[str] = []
        for q in base["questions"][sec]:
            if total >= 12:
                break
            trimmed.append(q)
            total += 1
        base["questions"][sec] = trimmed

    return base

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/generate", response_class=JSONResponse)
async def generate(
    dateISO: str = Form(...),
    respSupport: str = Form(...),
    feedingMethod: str = Form(...),
    weightKg: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "Missing OPENAI_API_KEY env var on server."}, status_code=500)

    dateISO = clamp_text(dateISO, 20)
    respSupport = clamp_text(respSupport, 30)
    feedingMethod = clamp_text(feedingMethod, 30)
    weightKg = clamp_text(weightKg, 20) if weightKg else None
    notes = clamp_text(notes, 1200) if notes else None

    if not notes and not weightKg and respSupport == "Room air" and feedingMethod == "Combo":
        return empty_payload()

    system = (
        "You are a supportive assistant helping parents of NICU babies prepare for medical rounds. "
        "You do NOT provide medical advice, diagnoses, or treatment recommendations. "
        "Your job is to rewrite parent notes into a clear neutral daily summary and generate respectful questions "
        "parents can ask their NICU care team. Encourage discussing concerns with clinicians."
    )

    developer = """
Return JSON only. No markdown. No extra text.

Structure:
{
  "summary": { "breathing": [], "feeding": [], "growth": [], "events": [] },
  "questions": { "breathing": [], "feeding": [], "growth": [], "events": [], "discharge": [] }
}

Rules:
- Max 12 questions total across all categories
- Questions must be neutral and non-directive (no "you should...")
- No medical advice, diagnosis, or treatment recommendations
- Keep bullets concise (<= 18 words each)
- If info is missing, keep lists empty rather than guessing
""".strip()

    user = f"""
Today's info:
Date: {dateISO}
Respiratory support: {respSupport}
Feeding: {feedingMethod}
Weight (kg): {weightKg or "unknown"}

Parent notes:
{notes or "none"}
""".strip()

    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "developer", "content": developer},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        parsed = json.loads(content)
        safe = enforce_shape(parsed)
        return safe
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
