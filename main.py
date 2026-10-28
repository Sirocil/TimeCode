"""Таймкод — семантический поиск по видеолекциям. Бэкенд (FastAPI).

Запуск: uvicorn main:app --reload
"""
import json
import os
import sqlite3
import subprocess
import uuid
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
DATA = ROOT / "data"
VIDEOS = DATA / "videos"
VIDEOS.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "app.db"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium
EMB_MODEL = os.getenv("EMB_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
LLM_KEY = os.getenv("OPENAI_API_KEY")  # любой OpenAI-совместимый API; без ключа — простой конспект
LLM_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

app = FastAPI(title="Таймкод")


# ---------- БД ----------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


with db() as c:
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS lectures(
          id TEXT PRIMARY KEY, title TEXT, status TEXT, error TEXT,
          video_path TEXT, duration REAL, transcript TEXT, summary TEXT, outline TEXT,
          created DATETIME DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY AUTOINCREMENT, lecture_id TEXT,
          start REAL, end REAL, text TEXT, emb BLOB);
        """
    )


def set_status(lid, status, **fields):
    fields["status"] = status
    sets = ", ".join(f"{k}=?" for k in fields)
    with db() as c:
        c.execute(f"UPDATE lectures SET {sets} WHERE id=?", [*fields.values(), lid])


# ---------- модели (ленивая загрузка) ----------
_whisper = _embedder = None


def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")
    return _whisper


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMB_MODEL)
    return _embedder


def embed(texts):
    return get_embedder().encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")


# ---------- обработка ----------
def fmt(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def make_chunks(segments, target=25.0):
    """Склеиваем соседние реплики в фрагменты ~25 с — это единицы поиска."""
    out, cur = [], None
    for s in segments:
        if cur is None:
            cur = dict(start=s["start"], end=s["end"], text=s["text"])
        else:
            cur["end"], cur["text"] = s["end"], cur["text"] + " " + s["text"]
        if cur["end"] - cur["start"] >= target:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def llm_outline(segments):
    import httpx

    transcript = "\n".join(f"[{fmt(s['start'])}] {s['text']}" for s in segments)[:60000]
    prompt = (
        "Ты помощник студента. По транскрипту лекции (с таймкодами) верни JSON вида "
        '{"summary": "краткий конспект 5-8 предложений", '
        '"sections": [{"title": "...", "start": секунды_числом}]} — 5-12 разделов по порядку. '
        "Только JSON, без пояснений.\n\n" + transcript
    )
    r = httpx.post(
        f"{LLM_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_KEY}"},
        json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
              "response_format": {"type": "json_object"}},
        timeout=120,
    )
    r.raise_for_status()
    data = json.loads(r.json()["choices"][0]["message"]["content"])
    return data["summary"], data["sections"]


def simple_outline(segments, duration):
    """Запасной вариант без LLM: делим на ~6 частей, заголовок — первая фраза."""
    n = 6 if duration > 600 else 3
    step = max(duration / n, 1)
    sections, summary = [], []
    for i in range(n):
        part = [s for s in segments if i * step <= s["start"] < (i + 1) * step]
        if part:
            sections.append({"title": part[0]["text"][:70], "start": part[0]["start"]})
            summary.append(part[0]["text"])
    return " ".join(summary), sections


def process(lid: str, source_path: str):
    try:
        set_status(lid, "extracting")
        audio = str(DATA / f"{lid}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", source_path, "-vn", "-ac", "1", "-ar", "16000", audio],
            check=True, capture_output=True,
        )

        set_status(lid, "transcribing")
        segs, info = get_whisper().transcribe(audio, vad_filter=True)
        segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs if s.text.strip()]
        duration = info.duration
        os.remove(audio)

        set_status(lid, "indexing", transcript=json.dumps(segments, ensure_ascii=False), duration=duration)
        chunks = make_chunks(segments)
        if chunks:
            embs = embed([c["text"] for c in chunks])
            with db() as c:
                c.executemany(
                    "INSERT INTO chunks(lecture_id,start,end,text,emb) VALUES(?,?,?,?,?)",
                    [(lid, ch["start"], ch["end"], ch["text"], embs[i].tobytes()) for i, ch in enumerate(chunks)],
                )

        set_status(lid, "summarizing")
        try:
            summary, sections = llm_outline(segments) if LLM_KEY else simple_outline(segments, duration)
        except Exception:
            summary, sections = simple_outline(segments, duration)
        set_status(lid, "ready", summary=summary, outline=json.dumps(sections, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        set_status(lid, "error", error=str(e)[:500])


# ---------- API ----------
@app.post("/api/lectures")
async def add_lecture(
    background: BackgroundTasks,
    title: str = Form(""),
    url: str = Form(""),
    file: UploadFile | None = File(None),
):
    lid = uuid.uuid4().hex[:10]
    if file and file.filename:
        ext = Path(file.filename).suffix.lower() or ".mp4"
        path = VIDEOS / f"{lid}{ext}"
        with open(path, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        title = title or Path(file.filename).stem
    elif url:
        try:
            import yt_dlp  # noqa: WPS433

            opts = {"outtmpl": str(VIDEOS / f"{lid}.%(ext)s"), "format": "mp4/best", "quiet": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                path = Path(ydl.prepare_filename(info))
                title = title or info.get("title", url)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Не удалось скачать видео по ссылке: {e}")
    else:
        raise HTTPException(400, "Нужен файл или ссылка")

    with db() as c:
        c.execute("INSERT INTO lectures(id,title,status,video_path) VALUES(?,?,?,?)",
                  (lid, title, "queued", str(path)))
    background.add_task(process, lid, str(path))
    return {"id": lid}


@app.get("/api/lectures")
def list_lectures():
    with db() as c:
        rows = c.execute("SELECT id,title,status,error,duration,created FROM lectures ORDER BY created DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/lectures/{lid}")
def get_lecture(lid: str):
    with db() as c:
        r = c.execute("SELECT * FROM lectures WHERE id=?", (lid,)).fetchone()
    if not r:
        raise HTTPException(404, "Лекция не найдена")
    d = dict(r)
    d["transcript"] = json.loads(d["transcript"] or "[]")
    d["outline"] = json.loads(d["outline"] or "[]")
    d.pop("video_path")
    return d


@app.delete("/api/lectures/{lid}")
def delete_lecture(lid: str):
    with db() as c:
        r = c.execute("SELECT video_path FROM lectures WHERE id=?", (lid,)).fetchone()
        c.execute("DELETE FROM chunks WHERE lecture_id=?", (lid,))
        c.execute("DELETE FROM lectures WHERE id=?", (lid,))
    if r and os.path.exists(r["video_path"]):
        os.remove(r["video_path"])
    return {"ok": True}


@app.get("/video/{lid}")
def video(lid: str):
    with db() as c:
        r = c.execute("SELECT video_path FROM lectures WHERE id=?", (lid,)).fetchone()
    if not r:
        raise HTTPException(404)
    return FileResponse(r["video_path"])  # поддерживает Range → перемотка работает


@app.get("/api/search")
def search(q: str, lecture_id: str | None = None, limit: int = 10):
    q = q.strip()
    if not q:
        return []
    sql = ("SELECT ch.lecture_id, ch.start, ch.end, ch.text, ch.emb, l.title "
           "FROM chunks ch JOIN lectures l ON l.id = ch.lecture_id")
    args = []
    if lecture_id:
        sql += " WHERE ch.lecture_id=?"
        args.append(lecture_id)
    with db() as c:
        rows = c.execute(sql, args).fetchall()
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(r["emb"], dtype="float32") for r in rows])
    scores = matrix @ embed([q])[0]  # косинусная близость (векторы нормализованы)
    top = np.argsort(-scores)[:limit]
    return [
        {
            "lecture_id": rows[i]["lecture_id"], "title": rows[i]["title"],
            "start": rows[i]["start"], "end": rows[i]["end"],
            "text": rows[i]["text"], "score": round(float(scores[i]), 3),
        }
        for i in top
    ]


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
