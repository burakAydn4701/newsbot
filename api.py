import json
import requests
import torch
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from search import search
from config import EMBED_MODEL, TOP_K, MAX_HISTORY

TGI_URL = "http://localhost:8001/v1/chat/completions"
MODEL   = "tgi"

# ── Model ─────────────────────────────────────────────────────────────────────
device      = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL, device=device)
print(f"[INFO] Embedding model loaded on {device}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Haberler AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# ── Prompt builder ────────────────────────────────────────────────────────────
def build_messages(history: list[Message], news_context: str) -> list[dict]:
    today = datetime.now().strftime("%d %B %Y, %A")

    system_prompt = f"""Sen haberler.com'un yapay zeka destekli haber asistanısın. Bugünün tarihi: {today}

Aşağıdaki HABERLER bölümündeki metinleri dikkatlice oku. Kullanıcının sorusunu YALNIZCA bu metinlerdeki bilgilere dayanarak yanıtla.

KURALLAR:
- Haberlerde geçen isimleri, tarihleri, sayıları ve gelişmeleri mutlaka cevabına dahil et.
- Cevabın en az 3-4 cümle olsun. Her 2-3 cümlede bir paragraf başlat (boş satır bırak).
- Kısa ve yetersiz cevap verme.
- Haberlerde yeterli bilgi yoksa sadece şunu yaz: "Bu konuda elimde güncel bir haber yok."
- Haber metinlerinin dışında hiçbir bilgi veya URL üretme.
- Kullanıcı haber dışı saçma bir şey yazarsa: "Sadece haberlerle ilgili sorulara yardımcı olabilirim." de.
- Kullanıcı selamlaşırsa kısa ve samimi karşılık ver, haberlere yönlendir.
- Cevabını paragraflara bölerek yaz.

HABERLER:
{news_context}"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    return messages

# ── SSE stream generator ──────────────────────────────────────────────────────
def stream_tgi(messages: list[dict]):
    try:
        with requests.post(
            TGI_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": True,
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            stream=True,
            timeout=300,
        ) as r:
            for line in r.iter_lines():
                if line:
                    line = line.decode()
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        data  = json.loads(line)
                        token = data["choices"][0]["delta"].get("content", "")
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except (json.JSONDecodeError, KeyError):
                        continue
    except requests.exceptions.RequestException as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    history = req.messages[-MAX_HISTORY:]

    search_query = next(
        (m.content for m in reversed(history) if m.role == "user"), ""
    )

    news_context, source_urls = search(search_query, embed_model, top_k=TOP_K)
    messages     = build_messages(history, news_context)

    def stream_with_sources():
        full_response = []
        for chunk in stream_tgi(messages):
            full_response.append(chunk)
            yield chunk
        full_text = "".join(full_response)
        no_news = any(p in full_text for p in ["elimde", "haber yok", "yardımcı olamıyorum", "ilgili haber", "güncel haber yok", "güncel bir haber yok"])
        if source_urls and not no_news:
            yield f"data: {json.dumps({'sources': source_urls})}\n\n"

    return StreamingResponse(
        stream_with_sources(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )

@app.get("/health")
async def health():
    return {"status": "ok"}