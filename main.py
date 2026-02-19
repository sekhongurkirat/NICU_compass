import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

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
    cgaWeeks: Optional[str] = Form(None),
    cgaDays: Optional[str] = Form(None),
    weightHistory: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "Missing OPENAI_API_KEY env var on server."}, status_code=500)

    dateISO = clamp_text(dateISO, 20)
    respSupport = clamp_text(respSupport, 30)
    feedingMethod = clamp_text(feedingMethod, 30)
    weightKg = clamp_text(weightKg, 20) if weightKg else None
    weightHistory = clamp_text(weightHistory, 300) if weightHistory else None
    notes = clamp_text(notes, 1200) if notes else None

    cga: Optional[str] = None
    if cgaWeeks:
        try:
            w = int(cgaWeeks)
            d = int(cgaDays) if cgaDays else 0
            if 22 <= w <= 44 and 0 <= d <= 6:
                cga = f"{w}+{d}"
        except ValueError:
            pass

    if not notes and not weightKg and respSupport == "Room air" and feedingMethod == "Combo":
        return empty_payload()

    system = """
You are a knowledgeable NICU family support assistant helping parents prepare for medical rounds.
You do NOT provide medical advice, diagnoses, or treatment recommendations.

Your two jobs:
1. SUMMARY — Rewrite the parent's raw notes into a clear, neutral, factual snapshot of today's status.
2. QUESTIONS — Generate a small set of high-quality questions the parent can ask the care team during rounds.

What makes a HIGH-QUALITY summary bullet:
- Factual and specific — includes actual values (e.g. "O2 weaned from 35% to 28%", not "oxygen is improving")
- One idea per bullet — do not combine unrelated facts
- Neutral tone — no reassurance, no alarm, just facts
- If weight history is provided, include a trend observation in the growth summary (e.g. "Weight has increased by 70g over the past 6 days")

What makes a HIGH-QUALITY rounds question:
- Open-ended and plan-oriented: starts with "What is the plan for...", "How will the team decide when...", "What would need to change for..."
- Grounded in today's specific data — references the actual respiratory support, feeds, weight, or events reported
- Helps the parent understand the next step or threshold, not just current status
- Appropriate for the baby's corrected gestational age:
  - < 28 weeks: focus on stability, brain protection, lung development
  - 28–32 weeks: focus on respiratory weaning, feed advancement, infection watch
  - 32–35 weeks: focus on oral feeding readiness, thermoregulation, weight gain rate
  - > 35 weeks: focus on discharge criteria, car seat test, home feeding plan
- Use the discharge category only when CGA > 34 weeks OR baby is on room air with oral feeds progressing
- Prioritised: the most urgent or parent-relevant question comes first

Example of a WEAK question (do not generate these):
- "Is the baby doing okay on the breathing support?"
- "Are the feeds going well?"

Example of a STRONG question (aim for these):
- "What FiO₂ threshold would the team want to see before trialling room air?"
- "At what daily feed volume would you consider transitioning from NG tube to full oral feeds?"
- "What is causing the overnight brady episodes and what is the plan if they continue?"
- "Given the weight has increased by 70g over 6 days, is the current growth rate on track for this gestational age?"

Generate 4–7 questions total. Fewer, sharper questions are better than many generic ones.
Only generate questions for categories where there is relevant data. Leave others empty.
""".strip()

    developer = """
Return JSON only. No markdown. No extra text.

Structure:
{
  "summary": { "breathing": [], "feeding": [], "growth": [], "events": [] },
  "questions": { "breathing": [], "feeding": [], "growth": [], "events": [], "discharge": [] }
}

Rules:
- 4–7 questions total across all categories, ordered by urgency/relevance
- Every question must directly reference a specific data point from today's info or parent notes
- Questions must be open-ended and plan-oriented (start with "What", "How", "When", "Which")
- No yes/no questions, no medical advice, no treatment recommendations
- Use the corrected gestational age to calibrate which topics matter most
- If weight history is provided, use it in the growth summary and to frame growth questions around the observed trend
- Only populate the discharge category if CGA > 34 weeks or baby is on room air with oral feeds progressing
- Keep each bullet concise (≤ 20 words)
- If a category has no relevant data, leave its list empty — do not guess or pad
""".strip()

    user = f"""
Today's info:
Date: {dateISO}
Corrected gestational age: {cga + " weeks" if cga else "unknown"}
Respiratory support: {respSupport}
Feeding: {feedingMethod}
Weight (kg): {weightKg or "unknown"}
Weight history: {weightHistory or "none"}

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
