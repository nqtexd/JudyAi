import json
import math
import os
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlparse

LOCAL_LIBS = Path(__file__).resolve().parent / ".pythonlibs"
if LOCAL_LIBS.exists():
    sys.path.insert(0, str(LOCAL_LIBS))

import requests
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - fallback for older installs
    try:
        from PyPDF2 import PdfReader
    except Exception:
        PdfReader = None


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "JudyAI-main"
DATA_DIR = ROOT / "data" / "cases"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_env_file(path):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(ROOT / ".env")
load_env_file(ROOT / ".env.local")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
LEGAL_SEARCH_BASE_URL = "https://duckduckgo.com/html/"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "80")) * 1024 * 1024


def cors_response(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


app.after_request(cors_response)


@app.route("/api/<path:_>", methods=["OPTIONS"])
def options(_):
    return ("", 204)


def case_dir(case_id):
    path = DATA_DIR / secure_filename(case_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "files").mkdir(exist_ok=True)
    return path


def metadata_path(case_id):
    return case_dir(case_id) / "case.json"


def read_case(case_id):
    path = metadata_path(case_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_case(case):
    metadata_path(case["id"]).write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def extract_pdf_text(pdf_path):
    if PdfReader is None:
        raise RuntimeError("PDF support is not installed. Run: pip install -r requirements.txt")
    reader = PdfReader(str(pdf_path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text = ""
        if text:
            pages.append({"page": index, "text": text})
    return pages


def chunk_pages(file_name, pages, chunk_words=420, overlap=80):
    chunks = []
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            part = " ".join(words[start : start + chunk_words])
            if part:
                chunks.append(
                    {
                        "id": f"{file_name}:p{page['page']}:{start}",
                        "file": file_name,
                        "page": page["page"],
                        "text": part,
                    }
                )
            start += max(1, chunk_words - overlap)
    return chunks


STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "has", "are", "was", "were",
    "shall", "would", "there", "their", "case", "court", "section", "act", "under", "into",
}


def terms(text):
    return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 2 and t not in STOPWORDS]


def retrieve_chunks(chunks, query, limit=7):
    query_terms = Counter(terms(query))
    if not query_terms:
        return chunks[:limit]
    scored = []
    total = max(1, len(chunks))
    doc_freq = Counter()
    tokenized = []
    for chunk in chunks:
        token_set = set(terms(chunk["text"]))
        tokenized.append(token_set)
        doc_freq.update(token_set)
    for chunk, token_set in zip(chunks, tokenized):
        score = 0.0
        for term, weight in query_terms.items():
            if term in token_set:
                score += weight * (1 + math.log(total / (1 + doc_freq[term])))
        if score:
            scored.append((score, chunk))
    return [chunk for _, chunk in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]] or chunks[:limit]


def call_groq(messages, temperature=0.15, max_tokens=3000, fallback_content=None):
    if not GROQ_API_KEY:
        return {
            "content": fallback_content or "The AI model is not configured yet, but the PDF retrieval layer is available.",
            "configured": False,
        }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return {"content": data["choices"][0]["message"]["content"], "configured": True}


def legal_search_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }


def strip_html(value):
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = (
        text.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return clean_text(text)


def normalize_search_url(url):
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return url


def legal_web_search(query, limit=5):
    if not query:
        return []
    try:
        scoped_query = (
            f"{query[:350]} Indian law case judgment statute "
            "site:sci.gov.in OR site:main.sci.gov.in OR site:indiacode.nic.in"
        )
        response = requests.get(
            LEGAL_SEARCH_BASE_URL,
            params={"q": scoped_query},
            headers=legal_search_headers(),
            timeout=25,
        )
        response.raise_for_status()
        html = response.text
        results = []
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
            r'(?:<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|'
            r'<div[^>]+class="result__snippet"[^>]*>(?P<snippet_div>.*?)</div>)',
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(html):
            title = strip_html(match.group("title"))
            url = normalize_search_url(match.group("url").replace("&amp;", "&"))
            snippet = strip_html(match.group("snippet") or match.group("snippet_div") or "")
            if title and url:
                results.append({"title": title, "snippet": snippet, "url": url})
            if len(results) >= limit:
                break
        if results:
            return results
        return [
            {
                "title": "No legal web leads found",
                "snippet": "The PDF analysis is still available, but the public legal search did not return usable results for this query.",
                "url": "",
                "unavailable": True,
            }
        ]
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        return [
            {
                "title": "Legal web search unavailable",
                "snippet": f"The public legal search returned HTTP {status or 'error'}. The PDF analysis is still available.",
                "url": "",
                "unavailable": True,
            }
        ]
    except Exception as exc:
        return [
            {
                "title": "Legal web search unavailable",
                "snippet": f"{exc}. The PDF analysis is still available.",
                "url": "",
                "unavailable": True,
            }
        ]


def first_case_query(case):
    text = " ".join(chunk["text"][:500] for chunk in case.get("chunks", [])[:5])
    title = case.get("title", "")
    return clean_text(f"{title} {text}")[:500]


def build_context(chunks):
    lines = []
    for chunk in chunks:
        lines.append(f"[{chunk['file']} page {chunk['page']}] {chunk['text']}")
    return "\n\n".join(lines)


def build_context_rich(chunks):
    """Richer context builder that groups chunks by file/page and adds
    positional hints so the model can detect cross-page contradictions."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"--- EXCERPT {i} | File: {chunk['file']} | Page: {chunk['page']} ---\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(lines)


def excerpt(text, words=44):
    parts = clean_text(text).split()
    if len(parts) <= words:
        return " ".join(parts)
    return " ".join(parts[:words]) + "..."


def local_analysis_fallback(case, chunks, legal_results):
    citations = "\n".join(
        f"- {chunk['file']} p.{chunk['page']}: {excerpt(chunk['text'])}" for chunk in chunks[:8]
    ) or "- No readable PDF excerpts were available."
    leads = "\n".join(
        f"- {lead['title']}: {lead.get('snippet') or lead.get('url') or 'No snippet available.'}"
        for lead in legal_results[:5]
    ) or "- No legal web leads returned."
    return (
        "## Executive Summary\n\n"
        f"Groq is not configured. This is a local PDF-first brief for **{case['title']}**. "
        "Add `GROQ_API_KEY` to `.env` and restart for full AI analysis.\n\n"
        "## Key Statutory Sections & Case Law\n\n"
        "The local fallback cannot infer statutory sections reliably beyond what appears in the uploaded PDFs. "
        "Inspect the cited excerpts below for section references.\n\n"
        "## Legal Web Leads\n\n"
        f"{leads}\n\n"
        "## Contradictions & Inconsistencies\n\n"
        "Full contradiction detection requires Groq. Excerpts for manual review:\n\n"
        f"{citations}\n\n"
        "## Missing Facts & Evidence Gaps\n\n"
        "1. Which facts are disputed by the opposing party?\n"
        "2. Which PDF pages contain the strongest evidence?\n"
        "3. Are there timeline discrepancies between witness statements?\n\n"
        "## Strategic Insights & Next Steps\n\n"
        "1. Verify statutory provisions against primary sources (indiacode.nic.in).\n"
        "2. Cross-reference cited judgments on the Supreme Court portal (sci.gov.in).\n"
        "3. Prepare a discovery strategy targeting the evidentiary gaps identified above."
    )


def local_chat_fallback(question, chunks, legal_results):
    citations = "\n".join(
        f"- {chunk['file']} p.{chunk['page']}: {excerpt(chunk['text'])}" for chunk in chunks[:6]
    ) or "- No matching PDF excerpts were found."
    leads = "\n".join(
        f"- {lead['title']}: {lead.get('snippet') or lead.get('url') or 'No snippet available.'}"
        for lead in legal_results[:4]
    ) or "- No legal web leads returned."
    return (
        "**Groq not configured** — showing local PDF-based response only.\n\n"
        f"**Question:** {question}\n\n"
        "**Most relevant PDF excerpts:**\n"
        f"{citations}\n\n"
        "**Legal web leads:**\n"
        f"{leads}\n\n"
        "Add `GROQ_API_KEY` to `.env` and restart the backend for full AI reasoning."
    )


def analyze_case(case):
    chunks = case.get("chunks", [])
    query = first_case_query(case)
    # Retrieve more chunks so the model has full coverage to find all contradictions
    retrieved = retrieve_chunks(chunks, query or case["title"], limit=18)
    legal_results = legal_web_search(query, limit=5)
    legal_context = "\n".join(f"- {r['title']}: {r['snippet']} ({r['url']})" for r in legal_results)
    context = build_context_rich(retrieved)

    system_prompt = (
        "You are JudyAI, a precise and thorough Indian legal research assistant. "
        "You MUST produce a complete, well-structured analysis strictly from the PDF excerpts provided.\n\n"
        "CRITICAL RULES:\n"
        "1. Read EVERY excerpt carefully from top to bottom before writing your response.\n"
        "2. Under 'Contradictions & Inconsistencies': find and list EVERY contradiction, discrepancy, "
        "or inconsistency you can identify — between witness statements, dates, amounts, facts, timelines, "
        "or exhibits. Number each one (1., 2., 3., ...). NEVER truncate the list. If there are 8 "
        "contradictions, list all 8. For each one, quote the conflicting text with page references.\n"
        "3. Under 'Key Statutory Sections & Case Law': cite specific Indian statutes (IPC, CrPC, CPC, "
        "IEA, specific Acts) and Supreme Court / High Court precedents that are directly relevant.\n"
        "4. Under 'Strategic Insights & Next Steps': give actionable, case-specific advice — not generic tips.\n"
        "5. Flag any uncertainty with [UNCERTAIN] inline.\n"
        "6. Use only the supplied excerpts and web results. Do not invent facts."
    )

    user_prompt = (
        f"**Case:** {case['title']}\n\n"
        f"**PDF EXCERPTS (read all carefully before responding):**\n\n{context}\n\n"
        f"**LEGAL WEB SEARCH RESULTS:**\n{legal_context or 'No results available.'}\n\n"
        "Produce your full analysis using EXACTLY these markdown headings in this order. "
        "Do not omit any heading. Be thorough and complete under each one:\n\n"
        "## Executive Summary\n"
        "(2-4 sentence overview of the case facts, key parties, core legal dispute)\n\n"
        "## Key Statutory Sections & Case Law\n"
        "(List each relevant Indian statute section and case precedent with a brief reason why it applies)\n\n"
        "## Legal Web Leads\n"
        "(Summarise the web search results and their relevance to this case)\n\n"
        "## Contradictions & Inconsistencies\n"
        "(Number EVERY contradiction found: 1. 2. 3. ... Quote conflicting text with page refs. "
        "List ALL of them — do not stop early.)\n\n"
        "## Missing Facts & Evidence Gaps\n"
        "(What key facts, documents, or witness statements are absent or incomplete)\n\n"
        "## Strategic Insights & Next Steps\n"
        "(Concrete, case-specific legal strategy recommendations)"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    groq = call_groq(
        messages,
        temperature=0.15,
        max_tokens=3000,
        fallback_content=local_analysis_fallback(case, retrieved, legal_results),
    )
    return {
        "summary": groq["content"],
        "groq_configured": groq["configured"],
        "legal_leads": legal_results,
        "indian_kanoon": legal_results,
        "citations": [{"file": c["file"], "page": c["page"], "id": c["id"]} for c in retrieved],
        "generated_at": int(time.time()),
    }


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "code.html")


@app.route("/chat.html")
def chat_page():
    return send_from_directory(FRONTEND_DIR, "chat.html")


@app.route("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "groq_configured": bool(GROQ_API_KEY),
            "legal_search_configured": True,
        }
    )


@app.route("/api/cases", methods=["POST"])
def create_case():
    title = clean_text(request.form.get("title") or "")
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Upload at least one PDF file."}), 400
    if not title:
        title = clean_text(Path(files[0].filename or "Untitled Case").stem.replace("_", " ").replace("-", " "))

    case_id = uuid.uuid4().hex
    directory = case_dir(case_id)
    stored_files = []
    chunks = []
    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            return jsonify({"error": f"{upload.filename} is not a PDF."}), 400
        safe_name = secure_filename(upload.filename) or f"{uuid.uuid4().hex}.pdf"
        destination = directory / "files" / safe_name
        upload.save(destination)
        pages = extract_pdf_text(destination)
        file_chunks = chunk_pages(safe_name, pages)
        chunks.extend(file_chunks)
        stored_files.append(
            {
                "name": safe_name,
                "size": destination.stat().st_size,
                "pages": len(pages),
                "chunks": len(file_chunks),
            }
        )

    case = {
        "id": case_id,
        "title": title,
        "status": "active",
        "date": int(time.time()),
        "files": stored_files,
        "chunks": chunks,
        "analysis": None,
    }
    case["analysis"] = analyze_case(case)
    write_case(case)
    public_case = {k: v for k, v in case.items() if k != "chunks"}
    return jsonify(public_case), 201


@app.route("/api/cases/<case_id>")
def get_case(case_id):
    case = read_case(case_id)
    if not case:
        return jsonify({"error": "Case not found."}), 404
    return jsonify({k: v for k, v in case.items() if k != "chunks"})


@app.route("/api/cases/<case_id>/chat", methods=["POST"])
def chat(case_id):
    case = read_case(case_id)
    if not case:
        return jsonify({"error": "Case not found."}), 404
    question = clean_text((request.get_json(silent=True) or {}).get("message", ""))
    if not question:
        return jsonify({"error": "Message is required."}), 400
    # Retrieve more chunks for richer context in chat responses
    retrieved = retrieve_chunks(case.get("chunks", []), question, limit=12)
    legal_results = legal_web_search(question, limit=4)
    legal_context = "\n".join(f"- {r['title']}: {r['snippet']} ({r['url']})" for r in legal_results)
    context = build_context_rich(retrieved)

    system_prompt = (
        "You are JudyAI, an expert Indian legal assistant specialising in case file analysis. "
        "You answer questions strictly from the uploaded PDF excerpts provided.\n\n"
        "RULES:\n"
        "1. Ground every claim in the provided excerpts. Quote relevant text with page references like (p.X).\n"
        "2. If the question asks about contradictions or inconsistencies, find and NUMBER every single one "
        "you can identify in the excerpts — do not stop early or say 'among others'.\n"
        "3. Cite specific Indian statutes (IPC, CrPC, CPC, IEA, etc.) and relevant case law where applicable.\n"
        "4. If the excerpts don't contain enough information to answer, say so explicitly — do not invent facts.\n"
        "5. Structure your answer with clear bold headings for each sub-topic."
    )

    user_prompt = (
        f"**Case:** {case['title']}\n"
        f"**Question:** {question}\n\n"
        f"**Relevant PDF excerpts:**\n\n{context}\n\n"
        f"**Legal web leads:**\n{legal_context or 'No results available.'}\n\n"
        "Answer the question thoroughly. If the question involves contradictions or inconsistencies, "
        "number and list ALL of them found in the excerpts above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    groq = call_groq(
        messages,
        temperature=0.15,
        max_tokens=2000,
        fallback_content=local_chat_fallback(question, retrieved, legal_results),
    )
    return jsonify(
        {
            "answer": groq["content"],
            "groq_configured": groq["configured"],
            "citations": [{"file": c["file"], "page": c["page"], "id": c["id"]} for c in retrieved],
            "legal_leads": legal_results,
            "indian_kanoon": legal_results,
        }
    )


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, use_reloader=False, port=int(os.getenv("PORT", "5000")))
