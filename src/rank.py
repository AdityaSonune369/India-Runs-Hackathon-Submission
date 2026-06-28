#!/usr/bin/env python3

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

DEFAULT_DATA_PATH = "data/candidates.jsonl"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

JD_TEXT = """
Senior AI Engineer — Founding Team at Redrob AI (Series A AI-native talent intelligence platform).
Location: Pune/Noida, India (Hybrid). Open to relocation from Tier-1 Indian cities.
Experience: roughly 5-9 years with production depth.

We need someone comfortable with BOTH:
- Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
- Scrappy product-engineering attitude — willing to ship a working ranker in a week even if suboptimal, learn from real users.

Mandate: own the intelligence layer (ranking, retrieval, and matching systems for recruiters and candidates).

Must have:
- Production experience with embeddings-based retrieval systems (sentence-transformers, BGE, E5, etc.) deployed to users.
- Production experience with vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS...).
- Strong Python and code quality.
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MAP, offline-to-online correlation, A/B).

Hackathon note: The right answer is NOT "candidates whose skills section contains the most AI keywords". That is a trap.
Reason about the gap between what the JD says and what it means. Heavily weigh behavioral signals.
"""

JD_TOKENS = [w.lower() for w in re.findall(r"\w+", JD_TEXT.lower())]

JD_KEY_PHRASES = [
    "embeddings", "retrieval", "ranking", "vector", "hybrid search", "faiss", "pinecone", "weaviate",
    "recommendation", "search system", "matching", "candidate", "recruiter", "production",
    "fine-tuning", "llm", "rag", "evaluation", "ndcg", "map", "a/b", "offline", "online",
    "ship", "deployed", "users", "scale", "pipeline", "python", "data", "ml", "ai engineer",
]

CONSULTING_FLAGS = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]

def get_candidate_text(cand):
    parts = []
    prof = cand.get("profile", {})
    parts.append(prof.get("headline", ""))
    parts.append(prof.get("summary", ""))
    for ch in cand.get("career_history", []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
        parts.append(ch.get("company", ""))
    for sk in cand.get("skills", []):
        parts.append(sk.get("name", ""))
    for edu in cand.get("education", []):
        parts.append(edu.get("institution", ""))
        parts.append(edu.get("field_of_study", ""))
        parts.append(edu.get("degree", ""))
    return " ".join(parts)

def compute_behavioral_score(cand):
    sig = cand.get("redrob_signals", {})
    if not sig:
        return 0.5
    score = 0.0
    if sig.get("open_to_work_flag"):
        score += 0.18
    rr = float(sig.get("recruiter_response_rate", 0.0) or 0.0)
    score += rr * 0.22
    views = min(sig.get("profile_views_received_30d", 0) or 0, 50) / 50.0
    score += views * 0.10
    saved = min(sig.get("saved_by_recruiters_30d", 0) or 0, 20) / 20.0
    score += saved * 0.08
    search_app = min(sig.get("search_appearance_30d", 0) or 0, 30) / 30.0
    score += search_app * 0.08
    last_active = sig.get("last_active_date")
    if last_active:
        try:
            days = (datetime.now() - datetime.fromisoformat(last_active)).days
            recency = max(0, 1 - min(days, 90) / 90.0)
            score += recency * 0.10
        except:
            pass
    completeness = (sig.get("profile_completeness_score", 50) or 50) / 100.0
    score += completeness * 0.06
    gh = sig.get("github_activity_score", -1) or -1
    if gh > 0:
        score += min(gh, 60) / 120.0
    icr = float(sig.get("interview_completion_rate", 0.0) or 0.0)
    score += icr * 0.08
    apps = sig.get("applications_submitted_30d", 0) or 0
    if apps == 0 and not sig.get("open_to_work_flag"):
        score -= 0.05
    return float(np.clip(score, 0.0, 1.0))

def compute_jd_overlap_score(cand):
    text_lower = get_candidate_text(cand).lower()
    skills = cand.get("skills", [])
    phrase_hits = sum(1 for p in JD_KEY_PHRASES if p in text_lower)
    phrase_score = min(phrase_hits / 6.0, 1.0) * 0.55
    skill_names_lower = [s.get("name", "").lower() for s in skills]
    skill_hits = sum(1 for p in JD_KEY_PHRASES if any(p in n for n in skill_names_lower))
    skill_score = min(skill_hits / 5.0, 1.0) * 0.25
    prof_bonus = 0.0
    for s in skills:
        if any(kw in s.get("name", "").lower() for kw in ["embed", "retriev", "rank", "vector", "llm", "fine", "python", "ml"]):
            prof = s.get("proficiency", "beginner")
            end = s.get("endorsements", 0) or 0
            if prof in ("advanced", "expert"):
                prof_bonus += 0.03 * min(end / 20.0, 1.0)
    prof_bonus = min(prof_bonus, 0.20)
    return float(np.clip(phrase_score + skill_score + prof_bonus, 0.0, 1.0))

def compute_career_fit(cand):
    career = cand.get("career_history", [])
    if not career:
        return 0.4
    score = 0.5
    text = " ".join([ch.get("description", "") + " " + ch.get("title", "") for ch in career]).lower()
    build_signals = ["built", "shipped", "deployed", "production", "pipeline", "system", "retrieval", "ranking", "search", "recommend"]
    build_count = sum(text.count(w) for w in build_signals)
    score += min(build_count / 8.0, 0.25)
    consulting_years = 0
    for ch in career:
        if any(c in ch.get("company", "").lower() for c in CONSULTING_FLAGS):
            consulting_years += ch.get("duration_months", 0) or 0
    if consulting_years > 36:
        score -= 0.25
    if "product" in text or "users" in text or "scale" in text:
        score += 0.10
    return float(np.clip(score, 0.1, 1.0))

def compute_exp_fit(cand):
    yrs = cand.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= yrs <= 9:
        return 1.0
    elif 4 <= yrs < 5 or 9 < yrs <= 11:
        return 0.85
    elif 3 <= yrs < 4 or 11 < yrs <= 13:
        return 0.65
    else:
        return max(0.3, 1.0 - abs(yrs - 7) * 0.08)

def compute_overall_score(cand, bm25_score, max_bm25):
    norm_bm25 = bm25_score / max(max_bm25, 1e-6)
    beh = compute_behavioral_score(cand)
    overlap = compute_jd_overlap_score(cand)
    career = compute_career_fit(cand)
    exp = compute_exp_fit(cand)
    final = (
        0.28 * norm_bm25 +
        0.32 * beh +
        0.18 * overlap +
        0.12 * career +
        0.10 * exp
    )
    return float(np.clip(final, 0.0, 1.0))

def generate_reasoning(cand, rank, score):
    prof = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})
    yrs = prof.get("years_of_experience", 0)
    title = prof.get("current_title", "Professional")
    skills = [s["name"] for s in cand.get("skills", [])[:5]]
    rr = sig.get("recruiter_response_rate", 0.0)
    base = f"{title} with {yrs:.1f} yrs; "
    if skills:
        base += f"key skills include {', '.join(skills[:3])}; "
    signals = []
    if sig.get("open_to_work_flag"):
        signals.append("open to work")
    if rr > 0.6:
        signals.append(f"high recruiter response {rr:.0%}")
    if sig.get("profile_views_received_30d", 0) > 8:
        signals.append("strong recent visibility")
    if sig.get("github_activity_score", -1) > 30:
        signals.append("github activity")
    if signals:
        base += "; ".join(signals) + "."
    notice = sig.get("notice_period_days", 0)
    if notice > 60 and rank < 30:
        base += f" Notice period {notice} days is a concern."
    if rank > 70:
        base += " Adjacent experience; lower priority vs stronger production retrieval matches."
    return base[:280]

def load_candidates(path):
    path = Path(path)
    is_gz = str(path).endswith(".gz") or str(path).endswith(".jsonl.gz")

    if is_gz:
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            start = ""
            for line in f:
                start = line.strip()
                if start:
                    break
    else:
        with open(path, "r", encoding="utf-8") as f:
            start = ""
            for line in f:
                start = line.strip()
                if start:
                    break

    if start.startswith("["):
        if is_gz:
            import gzip
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    records = []
    if is_gz:
        import gzip
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, "r", encoding="utf-8")
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def main():
    parser = argparse.ArgumentParser(
        description="India Runs Track 1 - Intelligent Candidate Discovery Ranker"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to candidates.jsonl (or .jsonl.gz). For judging, organizers provide this file."
    )
    parser.add_argument(
        "--out",
        default=str(OUTPUT_DIR / "submission.csv"),
        help="Output CSV path"
    )
    parser.add_argument("--topk", type=int, default=100, help="Number of candidates to output (must be 100 for final submission)")
    args = parser.parse_args()

    data_path = args.data

    if not os.path.exists(data_path):
        sample_p = os.path.join(os.path.dirname(data_path) or ".", "sample_candidates.json")
        if os.path.exists(sample_p):
            data_path = sample_p
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n\n"
                "For the official run, provide the path with --data (organizers will supply candidates.jsonl).\n"
                "Example: python -m src.rank --data /path/to/candidates.jsonl --out output/submission.csv\n"
                "For local testing, place sample_candidates.json next to the data path or in the current directory."
            )

    candidates = load_candidates(data_path)

    corpus = []
    for c in tqdm(candidates, desc="tokenizing"):
        corpus.append([w.lower() for w in re.findall(r"\w+", get_candidate_text(c))])

    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(JD_TOKENS)
    max_bm25 = float(np.max(bm25_scores)) or 1.0

    scored = []
    for i, cand in enumerate(tqdm(candidates, desc="scoring")):
        s = compute_overall_score(cand, bm25_scores[i], max_bm25)
        scored.append((s, cand))

    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:args.topk]
    rows = []
    for rank, (raw_score, cand) in enumerate(top, 1):
        score = round(0.99 - (rank - 1) * (0.99 - 0.40) / (args.topk - 1), 4)
        reason = generate_reasoning(cand, rank, score)
        rows.append({
            "candidate_id": cand["candidate_id"],
            "rank": rank,
            "score": score,
            "reasoning": reason
        })

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Wrote {len(df)} rows to {out_path}")
    print("Top 5:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
