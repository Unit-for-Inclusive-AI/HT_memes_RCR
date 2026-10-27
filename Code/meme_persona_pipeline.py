#!/usr/bin/env python3
"""
Persona-based multimodal meme classification with Reflect-Critique-Refine (RCR).

Expected classes:
  Homophobic, Transphobic, Non_Anti_LGBT

Example:
  pip install openai pandas openpyxl pillow requests scikit-learn
  export OPENAI_API_KEY="your_api_key"
  python meme_persona_pipeline.py --limit 1

The first run defaults to one image as a cost/sanity check. Remove --limit to process
all images after reviewing the output. The SoMe JSON schema is not assumed to be fixed:
the script recursively inspects JSON records and uses likely user/comment fields.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import hashlib
import sys
import zipfile
import io
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Automatically load local .env file (gitignored) if present
for _env_path in [Path(".env"), Path(__file__).resolve().parent.parent / ".env"]:
    if _env_path.exists():
        try:
            for _line in _env_path.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip('"').strip("'")
                    if _k not in os.environ:
                        os.environ[_k] = _v
        except Exception:
            pass

import pandas as pd
import requests
from openai import OpenAI
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(PROJECT_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from meme_pipeline.counter_narratives import generate_counter_narrative
from meme_pipeline.personas.selector import select_personas


DEFAULT_JSON_URL = (
    "https://raw.githubusercontent.com/LivXue/SoMe/main/"
    "datasets/user_comment_silmulation/ground_truth.json"
)
LABELS = ["Homophobic", "Transphobic", "Non_Anti_LGBT"]


def load_json_source(source: str) -> Any:
    """Load JSON from a local path or HTTP(S) URL."""
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        return response.json()
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


USER_KEYS = {
    "userid", "user", "username", "screenname", "author", "authorid",
    "accountid", "profileid", "handle", "useridstr"
}
COMMENT_KEYS = {
    "comment", "text", "content", "tweet", "post", "reply", "response",
    "commenttext", "tweettext", "posttext", "usercomment"
}
PROFILE_KEYS = {
    "profile", "userprofile", "description", "bio", "userdescription"
}
LANG_KEYS = {"language", "lang", "userlanguage"}


def first_value(record: dict, candidates: set[str]):
    for key, value in record.items():
        if norm_key(key) in candidates and isinstance(value, (str, int, float)):
            value = str(value).strip()
            if value:
                return value
    return ""


def walk_dicts(obj: Any):
    """Yield every dictionary nested in a JSON object."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_dicts(item)


def infer_language(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    if re.search(r"[\u0900-\u097f]", text):
        return "Hindi"
    return "English/other"


def style_summary(comments: list[str]) -> str:
    if not comments:
        return "No usable historical comments; use a neutral, cautious analytical style."
    avg_words = sum(len(x.split()) for x in comments) / len(comments)
    punctuation = sum(bool(re.search(r"[!?]{2,}|\.{3,}", x)) for x in comments)
    emoji_like = sum(bool(re.search(r"[😀-🙏]", x)) for x in comments)
    traits = []
    traits.append("usually brief" if avg_words < 12 else "often gives detailed comments")
    if punctuation / max(len(comments), 1) > 0.25:
        traits.append("uses emphatic punctuation")
    if emoji_like / max(len(comments), 1) > 0.2:
        traits.append("sometimes uses emoji")
    return "; ".join(traits) + ". This is a linguistic-style summary, not a claim about identity."


def extract_personas(data: Any, max_comments_per_user: int = 12) -> list[dict]:
    """
    Extract user/comment pairs from the actual SoMe ground_truth.json shape:
    {platform_user_id: [{"weibo_id": ..., "comment": ..., "label": ...}, ...]}.
    The source's `label` field is not used as a meme class or persona trait.
    Raw platform IDs are hashed in the saved persona pool.
    """
    grouped = defaultdict(lambda: {"comments": [], "profiles": [], "languages": Counter()})

    def add_record(user_id: str, rec: dict):
        comment = first_value(rec, COMMENT_KEYS)
        if not comment or len(comment) < 2 or len(comment) > 5000:
            return
        profile = first_value(rec, PROFILE_KEYS)
        lang = first_value(rec, LANG_KEYS) or infer_language(comment)
        grouped[str(user_id)]["comments"].append(comment)
        if profile:
            grouped[str(user_id)]["profiles"].append(profile)
        grouped[str(user_id)]["languages"][lang] += 1

    # Main supported format: each top-level key is a user ID and its value is a list
    # of comment records. This preserves each user's historical comment collection.
    if isinstance(data, dict):
        found_expected_shape = False
        for outer_user_id, value in data.items():
            if isinstance(value, list) and any(isinstance(x, dict) for x in value):
                records = [x for x in value if isinstance(x, dict)]
                if any(first_value(rec, COMMENT_KEYS) for rec in records):
                    found_expected_shape = True
                    for rec in records:
                        add_record(str(outer_user_id), rec)
        if not found_expected_shape:
            # Generic fallback for wrapped/nested JSON schemas.
            for idx, rec in enumerate(walk_dicts(data), start=1):
                comment = first_value(rec, COMMENT_KEYS)
                if not comment:
                    continue
                user_id = first_value(rec, USER_KEYS) or f"comment-derived-{idx}"
                add_record(user_id, rec)

    elif isinstance(data, list):
        for idx, rec in enumerate(walk_dicts(data), start=1):
            comment = first_value(rec, COMMENT_KEYS)
            if not comment:
                continue
            user_id = first_value(rec, USER_KEYS) or f"comment-derived-{idx}"
            add_record(user_id, rec)

    personas = []
    for idx, (user_id, values) in enumerate(grouped.items(), start=1):
        comments = values["comments"][:max_comments_per_user]
        profile_text = " ".join(values["profiles"][:3]).strip()
        language = values["languages"].most_common(1)[0][0] if values["languages"] else "Unknown"
        # Hash the platform ID to reduce unnecessary exposure in intermediate artifacts.
        pseudonymous_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
        personas.append({
            "persona_id": f"P{idx:05d}",
            "source_user_key_hash": pseudonymous_key,
            "language": language,
            "observed_profile_text": profile_text[:600],
            "historical_comments": comments,
            "commenting_style": style_summary(comments),
            "n_comments_found": len(values["comments"]),
            "evidence_note": (
                "Simulated persona derived from comments in the supplied JSON. The source's "
                "binary label field was not interpreted as an identity or meme-class label. "
                "Platform user ID is hashed. Attributes not explicitly present are not inferred."
            ),
        })

    return personas


def extract_image_id(filename: str):
    stem = Path(filename).stem
    match = re.search(r"\d+", stem)
    return int(match.group(0)) if match else None


def prepare_images(zip_path: str, image_dir: str):
    source_path = Path(zip_path)
    if source_path.is_dir():
        return source_path
    image_root = Path(image_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            suffix = Path(member).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                continue
            # Flatten files and use basename; image IDs are numeric in this dataset.
            target = image_root / Path(member).name
            if not target.exists():
                target.write_bytes(zf.read(member))
    return image_root


def load_image_label_map(xlsx_path: str) -> dict[int, str]:
    df = pd.read_excel(xlsx_path)
    columns = {str(column).strip().lower(): column for column in df.columns}
    id_column = next((columns[name] for name in ("id", "image_id") if name in columns), None)
    label_column = next((columns[name] for name in ("label", "class") if name in columns), None)
    if id_column is None or label_column is None:
        raise ValueError(
            "Excel must contain an ID column ('id' or 'image_id') and a label column "
            f"('label' or 'class'); found {list(df.columns)}"
        )
    result = {}
    for _, row in df.iterrows():
        if pd.isna(row[id_column]) or pd.isna(row[label_column]):
            continue
        label = str(row[label_column]).strip()
        # The English split uses noun forms while the debate schema uses adjective forms.
        label = {"Homophobia": "Homophobic", "Transphobia": "Transphobic"}.get(label, label)
        if label not in LABELS:
            raise ValueError(f"Unexpected label {label!r}. Expected one of {LABELS}")
        result[int(row[id_column])] = label
    return result


def image_to_data_url(path: Path) -> str:
    """Encode an image for a vision-capable OpenAI-compatible model."""
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif"
    }.get(suffix, "image/jpeg")
    # Convert GIF's first frame to PNG because some APIs don't accept GIF reliably.
    if suffix == ".gif":
        with Image.open(path) as im:
            im.seek(0)
            import io
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


class GeminiClientAdapter:
    """Expose Google GenAI through the provider-neutral pipeline helper."""

    _meme_pipeline_provider = "gemini"

    def __init__(self, api_key: str | None = None, project: str | None = None,
                 location: str = "global"):
        from google import genai

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_CLOUD_API_KEY")
        if project:
            # Use ADC for Vertex AI and Priority PayGo to reduce shared-capacity
            # throttling. Priority PayGo is billed at its premium token rate.
            from google.genai.types import HttpOptions

            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=HttpOptions(
                    api_version="v1",
                    headers={
                        "X-Vertex-AI-LLM-Request-Type": "shared",
                        "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
                    },
                ),
            )
        elif key:
            self._client = genai.Client(api_key=key)
        else:
            self._client = genai.Client(vertexai=True, location=location)
        self.models = self._client.models


def chat_text(client, model: str, system: str, user_content, temperature=0.2, max_retries: int = 5) -> str:
    import time

    for attempt in range(1, max_retries + 1):
        try:
            if getattr(client, "_meme_pipeline_provider", None) == "gemini":
                from google.genai import types

                contents = []
                for item in user_content if isinstance(user_content, list) else [user_content]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        contents.append(item.get("text", ""))
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        data_url = item["image_url"]["url"]
                        encoded = data_url.split(",", 1)[1]
                        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                            contents.append(image.copy())
                    else:
                        contents.append(str(item))
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                    ),
                )
                time.sleep(1.0)  # gentle pacing to avoid bursting Vertex AI QPS quota
                return response.text or ""

            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            )
            time.sleep(0.5)
            return response.choices[0].message.content or ""

        except Exception as e:
            err_msg = str(e)
            is_rate_limit = any(k in err_msg for k in ["429", "RESOURCE_EXHAUSTED", "ResourceExhausted", "RateLimit", "rate_limit"])
            if is_rate_limit and attempt < max_retries:
                wait_sec = 2 ** attempt * 3  # 6s, 12s, 24s, 48s backoff
                print(f"  [!] Rate limit/quota (429 RESOURCE_EXHAUSTED). Backing off for {wait_sec}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_sec)
            elif ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < max_retries:
                wait_sec = 5 * attempt
                print(f"  [!] Service temporarily unavailable (503). Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                raise


def image_content(image_url: str, text: str):
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
    ]


def safe_json(text: str) -> dict:
    """Parse a JSON object even if the model wraps it in Markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"raw_response": text}


def rank_personas(personas: list[dict], meme_text: str, k: int = 8) -> list[dict]:
    """Compatibility wrapper returning the scored persona shortlist."""
    candidates, _ = select_personas(personas, meme_text, shortlist_size=k, final_count=3)
    return candidates


def summarize_meme(client: OpenAI, model: str, image_url: str) -> dict:
    system = (
        "You are a careful multimodal research assistant analysing a meme for a research "
        "dataset about anti-LGBTQ+ content. Describe visible elements and transcribe readable "
        "text as accurately as possible. Do not assume a person's sexual orientation, gender "
        "identity, or intent from appearance alone. Distinguish explicit evidence from ambiguity. "
        "Return only JSON with keys: visible_text, visual_description, context_clues, uncertainty."
    )
    response = chat_text(
        client, model, system,
        image_content(image_url, "Inspect this meme. Transcribe visible text and describe relevant context."),
        temperature=0.0,
    )
    return safe_json(response)


def select_image_persona_candidates(client, model: str, personas: list[dict], meme_summary: dict,
                                   image_url: str, candidate_count: int,
                                   selection_seed: int = 0) -> list[dict]:
    """Build an image-specific shortlist with a stable, per-image persona order."""
    # Vary prompt positions by image to reduce repeated first-item selections while
    # keeping the same image reproducible if its run is retried.
    ordered_personas = sorted(
        personas,
        key=lambda p: hashlib.sha256(
            f"{selection_seed}:{p.get('persona_id', '')}".encode("utf-8")
        ).digest(),
    )
    council = []
    for persona in ordered_personas:
        council.append({
            "persona_id": persona.get("persona_id"),
            "profile": persona.get("profile_bio", persona.get("observed_profile_text", "")),
            "style": persona.get("commenting_habits", persona.get("commenting_style", "")),
            "historical_comments": persona.get("sample_historical_comments", persona.get("historical_comments", [])),
        })
    system = (
        f"Select exactly {candidate_count} candidate simulated personas for this specific meme. "
        "This is a broad shortlist for a separate downstream trio selection step; do not select "
        "only the final three. Choose candidates only when their supplied profile details, "
        "observable commenting style, or historical comments provide a plausible basis for "
        "contributing to this meme's interpretation, while including varied perspectives. Ground "
        "each reason in that evidence; do not invent interests or beliefs. Archetype labels are "
        "synthetic experimental labels and are deliberately not provided as evidence for this step. "
        "Evaluate each candidate independently; do not favor candidates because they appear earlier in the list. "
        "Do not infer protected traits or use or guess the dataset label. Return JSON only: "
        '{"candidates":[{"persona_id":"...","image_fit":"brief reason grounded in the supplied analysis and persona evidence"}]}'
    )
    prompt = json.dumps({"meme_analysis": meme_summary, "persona_council": council}, ensure_ascii=False)
    try:
        result = safe_json(chat_text(
            client, model, system, image_content(image_url, prompt), temperature=0.0
        ))
        selections = result.get("candidates", [])
        by_id = {str(p.get("persona_id")): p for p in personas}
        picked = []
        for selection in selections:
            if not isinstance(selection, dict):
                continue
            persona = by_id.get(str(selection.get("persona_id", "")))
            if persona and persona not in picked:
                item = dict(persona)
                item["image_fit"] = str(selection.get("image_fit", ""))
                picked.append(item)
            if len(picked) == candidate_count:
                return picked
    except Exception as exc:
        print(f"  [!] Image-specific candidate selection failed ({exc}); using scored fallback.")

    # If the model response is invalid, rank the complete pool against this image
    # so the fallback remains image-specific instead of taking the first IDs.
    ranking_text = " ".join(str(meme_summary.get(key, "")) for key in (
        "visible_text", "visual_description", "context_clues"
    ))
    print("  [!] Using deterministic image-relevance fallback to build persona candidates.")
    _, fallback = select_personas(
        ordered_personas, ranking_text, shortlist_size=candidate_count, final_count=candidate_count
    )
    return fallback


def classify_with_persona(client: OpenAI, model: str, image_url: str, meme_summary: dict,
                          persona: dict, peer_responses: list[dict] | None = None) -> dict:
    system = (
        "You are one simulated analytical persona in a research debate about meme classification. "
        "Your persona is a perspective-taking simulation, not a real person. Do not produce slurs or "
        "expand hateful content unnecessarily. Judge the meme, not the worth of a group. "
        "Allowed labels are exactly: Homophobic, Transphobic, Non_Anti_LGBT. "
        "Homophobic means the meme targets, demeans, stereotypes, excludes, or expresses hostility "
        "toward gay/lesbian/bisexual people or same-sex attraction. Transphobic means it targets, "
        "demeans, stereotypes, excludes, or expresses hostility toward transgender people or gender "
        "transition. Non_Anti_LGBT means the image does not clearly express anti-LGBTQ+ hostility, "
        "including supportive, neutral, unrelated, or insufficiently evidenced cases. "
        "Do not infer hate from LGBTQ+ identity alone. Consider text, image, context, sarcasm, and "
        "whether the target is actually an LGBTQ+ group. Return JSON only with keys label, confidence "
        "(0 to 1), evidence, ambiguity, and rationale."
    )
    history = persona.get("historical_comments", persona.get("sample_historical_comments", []))[:5]
    persona_info = {
        "persona_id": persona["persona_id"],
        "language": persona.get("language"),
        "commenting_style": persona.get("commenting_style") or persona.get("commenting_habits", {}).get("summary"),
        "source_profile_text": persona.get("observed_profile_text") or persona.get("profile_bio", ""),
        "historical_comment_examples": history,
        "evidence_note": persona.get("evidence_note"),
    }
    peers = peer_responses or []
    prompt = (
        f"Meme analysis:\n{json.dumps(meme_summary, ensure_ascii=False)}\n\n"
        f"Your simulated persona:\n{json.dumps(persona_info, ensure_ascii=False)}\n\n"
        f"Peer responses from the previous round (if any):\n{json.dumps(peers, ensure_ascii=False)}\n\n"
        "Classify this meme using the allowed labels. Explain only the evidence needed for the decision. "
        "If changing a previous position, identify a new piece of evidence or reasoning; do not change "
        "your answer merely to agree with peers."
    )
    response = chat_text(
        client, model, system,
        image_content(image_url, prompt),
        temperature=0.3,
    )
    parsed = safe_json(response)
    parsed["persona_id"] = persona["persona_id"]
    return parsed


def rcr_refine(client: OpenAI, model: str, image_url: str, meme_summary: dict,
               persona: dict, initial: dict, peers: list[dict], round_number: int = 1) -> dict:
    system = (
        "Follow Reflect-Critique-Refine (RCR). Reflect: identify a possible weakness in your initial "
        "classification. Critique: assess exactly two peer responses when available, identifying evidence "
        "or reasoning errors without following confidence or verbosity alone. Refine: keep or change your "
        "label only when evidence supports it. Do not infer identity or intent from appearance alone. "
        "Allowed labels: Homophobic, Transphobic, Non_Anti_LGBT. Return JSON only with keys: label, "
        "confidence, evidence, self_critique, peer_critiques, refinement_reason, uncertainty."
    )
    peer_pair = peers[:2]
    prompt = (
        f"Meme analysis: {json.dumps(meme_summary, ensure_ascii=False)}\n"
        f"Persona: {json.dumps({'persona_id': persona['persona_id'], 'language': persona.get('language'), 'style': persona.get('commenting_style')}, ensure_ascii=False)}\n"
        f"Your current response before RCR round {round_number}: {json.dumps(initial, ensure_ascii=False)}\n"
        f"Two peer responses to critique: {json.dumps(peer_pair, ensure_ascii=False)}\n"
        "Complete the three RCR phases and return a refined decision. On later rounds, "
        "critique the current response and the latest peer responses, not an older round."
    )
    response = chat_text(
        client, model, system,
        image_content(image_url, prompt),
        temperature=0.2,
    )
    parsed = safe_json(response)
    parsed["persona_id"] = persona["persona_id"]
    return parsed


def debate_consensus(responses: list[dict]) -> tuple[str | None, dict[str, int]]:
    votes = [item.get("label") for item in responses if item.get("label") in LABELS]
    counts = dict(Counter(votes))
    if not counts:
        return None, counts
    highest = max(counts.values())
    leaders = [label for label, count in counts.items() if count == highest]
    consensus = leaders[0] if highest >= 2 and len(leaders) == 1 else None
    return consensus, counts


def critic_judge(client: OpenAI, model: str, image_url: str, meme_summary: dict,
                 initial: list[dict], refined: list[dict]) -> dict:
    system = (
        "You are the independent critic/judge for a research classification pipeline. Compare the "
        "agents' initial and RCR-refined answers. Check whether each decision is grounded in visible "
        "text/context, whether agents over-inferred, and whether they simply copied peers. Choose one "
        "label from Homophobic, Transphobic, Non_Anti_LGBT. Do not force consensus if evidence is weak; "
        "use Non_Anti_LGBT when anti-LGBTQ+ hostility is not supported by evidence, and mark uncertainty. "
        "This is a dataset label, not a moral judgment about a person. Return JSON only with keys: label, "
        "confidence, evidence, disagreement_summary, uncertainty, agent_vote_counts, final_reason."
    )
    prompt = (
        f"Meme analysis: {json.dumps(meme_summary, ensure_ascii=False)}\n"
        f"Initial agent responses: {json.dumps(initial, ensure_ascii=False)}\n"
        f"RCR-refined agent responses: {json.dumps(refined, ensure_ascii=False)}\n"
        "Independently adjudicate the final dataset label. Prioritise evidence over majority alone."
    )
    response = chat_text(
        client, model, system,
        image_content(image_url, prompt),
        temperature=0.0,
    )
    result = safe_json(response)
    if result.get("label") not in LABELS:
        valid = [r.get("label") for r in refined if r.get("label") in LABELS]
        result["label"] = Counter(valid).most_common(1)[0][0] if valid else "Non_Anti_LGBT"
        result["fallback_note"] = "Judge did not return a valid label; used majority of valid refined labels."
    return result


def generate_counter_narrative_for_meme(
    client, model: str, image_url: str, meme_analysis: dict, classification: dict,
    fallback_language: str,
) -> dict:
    visible_text = str(meme_analysis.get("visible_text", ""))
    detected = infer_language(visible_text) if visible_text else "English/other"
    language = ({"Chinese": "Chinese", "Hindi": "Hindi"}.get(detected, "English")
                if visible_text else fallback_language)

    def generator(prompt: str) -> str:
        system = (
            "You create concise counterspeech for a research tool. Be supportive and respectful, "
            "address only harmful ideas supported by the supplied evidence, and do not repeat slurs. "
            "Return a JSON object with text and language."
        )
        return chat_text(client, model, system, image_content(image_url, prompt), temperature=0.2)

    return generate_counter_narrative(classification, meme_analysis, language, generator)


def main():
    parser = argparse.ArgumentParser(description="Persona debate pipeline for anti-LGBTQ+ meme classification.")
    parser.add_argument("--json-source", default=DEFAULT_JSON_URL,
                        help="SoMe ground_truth.json URL or local JSON path")
    parser.add_argument("--images-zip", default="Train_images-20261012T165402Z-1-001.zip",
                        help="Image archive or an existing directory of images")
    parser.add_argument("--labels-xlsx", default="Train_labels.xlsx")
    parser.add_argument("--image-dir", default="prepared_images")
    parser.add_argument("--output", default="meme_predictions.jsonl")
    parser.add_argument("--persona-output", default="persona_pool.json")
    parser.add_argument("--persona-pool", default=None,
                        help="Use an existing persona-pool JSON instead of extracting from SoMe")
    parser.add_argument("--model", default=None,
                        help="API model or local model ID (defaults depend on --backend)")
    parser.add_argument("--backend", choices=["openai", "gemini", "vertex", "local"], default="openai",
                        help="Use OpenAI, Gemini Developer API, Google Cloud Vertex AI, or local Transformers")
    parser.add_argument("--gcp-project", default=os.getenv("GOOGLE_CLOUD_PROJECT"),
                        help="Google Cloud project ID for --backend vertex")
    parser.add_argument("--gcp-location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
                        help="Vertex AI location for --backend vertex (default: global)")
    parser.add_argument("--adapter-path", default=None,
                        help="Optional trained PEFT adapter directory for --backend local")
    parser.add_argument("--rcr-rounds", type=int, default=1,
                        help="Maximum RCR debate rounds per meme (default preserves one-round behavior)")
    parser.add_argument("--evolution-round", type=int, default=None,
                        help="Optional evolution round number to record in traces")
    parser.add_argument("--hide-labels", action="store_true",
                        help="Do not load, store, or score workbook labels during trace generation")
    parser.add_argument("--consensus-only", action="store_true",
                        help="Use the three refined persona votes; call the judge if there is no clear majority")
    parser.add_argument("--skip-counter-narrative", action="store_true",
                        help="Skip optional counter-narrative generation")
    parser.add_argument("--agents", type=int, default=3, help="Final debate personas; DTE RCR uses 3")
    parser.add_argument("--candidate-count", type=int, default=6,
                        help="Personas to score before choosing the final trio")
    parser.add_argument("--language", choices=["Chinese", "English", "Hindi"], default="English",
                        help="Dataset language; visible meme text overrides this for counter-narratives")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Dataset split recorded in the output; only train traces are accepted by GRPO")
    parser.add_argument("--limit", type=int, default=1, help="Images to process; default 1 for cost testing; 0 means all")
    parser.add_argument("--image-id", type=int, default=None, help="Process one specific image ID")
    parser.add_argument("--max-personas", type=int, default=1000, help="Cap the extracted persona pool")
    args = parser.parse_args()

    if args.agents != 3:
        raise ValueError("This DTE RCR workflow uses exactly three debate personas.")
    if args.rcr_rounds < 1:
        raise ValueError("--rcr-rounds must be at least 1")
    if not 6 <= args.candidate_count <= 10:
        raise ValueError("--candidate-count must be between 6 and 10")
    model = args.model or os.getenv({"local": "LOCAL_MODEL", "gemini": "GEMINI_MODEL", "vertex": "VERTEX_MODEL", "openai": "OPENAI_MODEL"}[args.backend]) or {
        "local": "Qwen/Qwen2.5-VL-3B-Instruct",
        "gemini": "gemini-3.8-flash",
        "vertex": "gemini-3.8-flash",
        "openai": "gpt-4.1-mini",
    }[args.backend]
    if args.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running this script.")
    if args.backend == "gemini" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_CLOUD_API_KEY")):
        raise EnvironmentError("Set GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_CLOUD_API_KEY in .env before running this script.")
    if args.backend == "vertex" and not args.gcp_project:
        raise EnvironmentError("Set GOOGLE_CLOUD_PROJECT or pass --gcp-project for Vertex AI.")

    persona_pool_path = args.persona_pool
    if not persona_pool_path and args.language == "Chinese":
        for candidate_pool in (
            "artifacts/persona_pool_zh.json",
            "artifacts/persona_council_zh.json",
        ):
            if os.path.exists(candidate_pool):
                persona_pool_path = candidate_pool
                break

    if persona_pool_path:
        print(f"Loading persona pool from {persona_pool_path}...")
        with open(persona_pool_path, "r", encoding="utf-8") as f:
            personas = json.load(f)
        if not isinstance(personas, list):
            raise ValueError("Persona pool JSON must contain a list of persona objects.")
    else:
        print("Loading SoMe JSON...")
        data = load_json_source(args.json_source)
        personas = extract_personas(data)
    if not personas:
        print("WARNING: no user/comment records were detected in the JSON schema.")
        print("The input's top-level type is:", type(personas).__name__)
        if not args.persona_pool and isinstance(data, dict):
            print("Top-level keys:", list(data.keys())[:50])
        raise RuntimeError(
            "Could not build personas automatically. Inspect the printed schema and update "
            "USER_KEYS/COMMENT_KEYS in the script to match the JSON fields."
        )
    personas = personas[:args.max_personas]
    if not persona_pool_path:
        Path(args.persona_output).write_text(
            json.dumps(personas, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Extracted {len(personas)} simulated personas -> {args.persona_output}")
    else:
        print(f"Loaded {len(personas)} personas from {persona_pool_path}")

    labels = {} if args.hide_labels else load_image_label_map(args.labels_xlsx)
    image_root = prepare_images(args.images_zip, args.image_dir)
    image_files = {}
    for path in image_root.iterdir():
        if path.is_file():
            image_id = extract_image_id(path.name)
            if image_id is not None:
                if image_id in image_files:
                    raise ValueError(
                        f"Multiple image files map to ID {image_id}: "
                        f"{image_files[image_id].name!r} and {path.name!r}"
                    )
                image_files[image_id] = path

    ids = sorted(set(image_files) if args.hide_labels else set(labels) & set(image_files))
    if args.image_id is not None:
        ids = [i for i in ids if i == args.image_id]
        if not ids:
            raise ValueError(f"Image ID {args.image_id} was not found in both the labels and image archive.")
    elif args.limit > 0:
        ids = ids[:args.limit]

    if args.backend == "local":
        from meme_pipeline.models.local_vlm import LocalVisionLanguageClient

        client = LocalVisionLanguageClient.from_pretrained(model, args.adapter_path)
    elif args.backend == "gemini":
        client = GeminiClientAdapter(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_CLOUD_API_KEY"))
    elif args.backend == "vertex":
        client = GeminiClientAdapter(
            project=args.gcp_project,
            location=args.gcp_location
        )
    else:
        client = OpenAI()
    out_path = Path(args.output)
    already_done = set()
    if out_path.exists():
        existing_ids = []
        for line_number, line in enumerate(out_path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                image_id = int(json.loads(line)["image_id"])
            except Exception as exc:
                raise ValueError(
                    f"Existing output {out_path} has an invalid JSONL record on line {line_number}. "
                    "Move or delete the output before starting a clean run."
                ) from exc
            if image_id in already_done:
                raise ValueError(
                    f"Existing output {out_path} contains duplicate image ID {image_id}. "
                    "Move or delete the output before starting a clean run."
                )
            if existing_ids and image_id <= existing_ids[-1]:
                raise ValueError(
                    f"Existing output {out_path} is not in strictly increasing image_id order "
                    f"(line {line_number}: {image_id}). Move or delete it before restarting; "
                    "the pipeline will not append to a shuffled file."
                )
            existing_ids.append(image_id)
            already_done.add(image_id)

    with out_path.open("a", encoding="utf-8") as out:
        for n, image_id in enumerate(ids, start=1):
            if image_id in already_done:
                print(f"[{n}/{len(ids)}] Skipping already processed ID {image_id}")
                continue
            image_path = image_files[image_id]
            print(f"[{n}/{len(ids)}] Processing image ID {image_id} ({image_path.name})")
            image_url = image_to_data_url(image_path)

            meme_summary = summarize_meme(client, model, image_url)
            ranking_text = " ".join([
                str(meme_summary.get("visible_text", "")),
                str(meme_summary.get("visual_description", "")),
                str(meme_summary.get("context_clues", "")),
            ])
            image_candidate_pool = select_image_persona_candidates(
                client, model, personas, meme_summary, image_url, args.candidate_count,
                selection_seed=image_id,
            )
            candidates, selected = select_personas(
                image_candidate_pool, ranking_text,
                shortlist_size=args.candidate_count, final_count=3
            )
            if len(selected) < 3:
                raise RuntimeError("At least three personas are needed for DTE debate.")
            selected_ids = {persona.get("persona_id") for persona in selected}
            ranked_by_id = {p.get("persona_id"): p for p in candidates}
            fit_by_id = {p.get("persona_id"): p.get("image_fit", "") for p in selected}
            image_candidates = []
            for persona in image_candidate_pool:
                candidate = dict(ranked_by_id.get(persona.get("persona_id"), persona))
                candidate["selected_for_image"] = persona.get("persona_id") in selected_ids
                if candidate["selected_for_image"]:
                    candidate["image_fit"] = fit_by_id.get(persona.get("persona_id"), "")
                image_candidates.append(candidate)

            initial = []
            for persona in selected:
                initial.append(classify_with_persona(
                    client, model, image_url, meme_summary, persona
                ))

            current = initial
            debate_rounds = []
            refined = []
            consensus = None
            vote_counts = {}
            for round_number in range(1, args.rcr_rounds + 1):
                refined = []
                for i, persona in enumerate(selected):
                    peer_responses = [r for j, r in enumerate(current) if j != i]
                    refined.append(rcr_refine(
                        client, model, image_url, meme_summary,
                        persona, current[i], peer_responses, round_number=round_number
                    ))
                consensus, vote_counts = debate_consensus(refined)
                debate_rounds.append({
                    "round": round_number,
                    "responses": refined,
                    "vote_counts": vote_counts,
                    "consensus_reached": len(vote_counts) == 1 and sum(vote_counts.values()) == 3,
                })
                current = refined
                if len(vote_counts) == 1 and sum(vote_counts.values()) == 3:
                    break

            if args.consensus_only and consensus is not None:
                final = {
                    "label": consensus,
                    "confidence": None,
                    "evidence": [r.get("evidence") for r in refined if r.get("evidence")],
                    "disagreement_summary": "Persona vote consensus/majority after RCR.",
                    "uncertainty": None,
                    "agent_vote_counts": vote_counts,
                    "final_reason": "Label selected by majority vote of the three refined persona responses.",
                    "selection_method": "rcr_majority_vote",
                }
            else:
                final = critic_judge(client, model, image_url, meme_summary, initial, refined)
                final["selection_method"] = "independent_critic_judge"
            if consensus is None and final.get("label") in LABELS:
                consensus = final["label"]
                consensus_source = "critic_fallback_no_clear_majority"
            else:
                consensus_source = "persona_majority_vote"
            if args.skip_counter_narrative:
                counter_narrative = {"status": "skipped", "text": None, "language": args.language}
            else:
                counter_narrative = generate_counter_narrative_for_meme(
                    client, model, image_url, meme_summary, final, args.language
                )
            record = {
                "image_id": image_id,
                "image_file": image_path.name,
                "language": args.language,
                "split": args.split,
                "predicted_label": final.get("label"),
                "meme_analysis": meme_summary,
                "persona_candidates": image_candidates,
                "selected_personas": selected,
                "initial_debate": initial,
                "rcr_refined_responses": refined,
                "debate_rounds": debate_rounds,
                "consensus_label": consensus,
                "consensus_source": consensus_source,
                "consensus_vote_counts": vote_counts,
                "critic_judge": final,
                "counter_narrative": counter_narrative,
                "model": model,
                "adapter_path": args.adapter_path,
                "evolution_round": args.evolution_round,
                "persona_selection_method": "Two-stage: model selects six to ten image-specific candidates from the full council; deterministic relevance scoring uses each candidate's image-fit rationale and historical comments, with evidence, diversity, and complementarity scoring selecting the final three from that shortlist.",
                "method_note": "Gold labels are stored for evaluation and GRPO rewards only; they are not sent to persona selection, persona agents, or the counter-narrative generator.",
            }
            if not args.hide_labels:
                record["gold_label"] = labels[image_id]
                record["correct"] = final.get("label") == labels[image_id]
            else:
                record["method_note"] = (
                    "Labels were not loaded during trace generation; debate, persona selection, "
                    "and counter-narrative generation received no ground-truth labels."
                )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            persona_ids = ", ".join(persona["persona_id"] for persona in selected)
            print(
                f"  id={image_id} | personas={persona_ids} | final={final.get('label')} "
                f"| counter_narrative={counter_narrative.get('status')}"
            )

    print(f"Finished. Results appended to {out_path.resolve()}")

    # Evaluate every completed prediction that has a valid gold label.
    completed = []
    if args.hide_labels:
        print("Label loading and evaluation were disabled for this trace-generation run.")
        return
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if item.get("gold_label") in LABELS and item.get("predicted_label") in LABELS:
                completed.append(item)
        except Exception:
            continue
    if completed:
        y_true = [x["gold_label"] for x in completed]
        y_pred = [x["predicted_label"] for x in completed]
        report = {
            "n_evaluated": len(completed),
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
            ),
            "confusion_matrix_labels_order": LABELS,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
            "note": "Metrics are preliminary if only a small --limit sample was processed.",
        }
        metrics_path = out_path.with_name("evaluation_metrics.json")
        metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Evaluation metrics written to {metrics_path.resolve()}")
        print(f"Evaluated {len(completed)} predictions; accuracy={report['accuracy']:.3f}")

    print("Important: validate JSON field extraction, persona quality, and labels manually on a sample before scaling up.")


if __name__ == "__main__":
    main()
