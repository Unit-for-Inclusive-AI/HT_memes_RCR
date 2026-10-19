"""
Unified Persona Extractor for Chinese and English Social Data.
Path: src/meme_pipeline/personas/extractor.py
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

USER_KEYS = {"userid", "user", "username", "screenname", "author", "authorid", "accountid", "handle", "weibo_id"}
COMMENT_KEYS = {"comment", "text", "content", "weibo", "post", "reply", "commenttext", "usercomment"}
PROFILE_KEYS = {"profile", "userprofile", "description", "bio", "userdescription", "occupation", "intro"}
PINYIN_SLANG = {"cnm", "nmsl", "tmd", "nb", "sb", "woc", "yyds"}


def is_valid_comment(text: str, language: str) -> bool:
    text_clean = text.strip()
    if not text_clean:
        return False

    if language.lower() == "chinese":
        # Must contain Hanzi and be at least 4 characters long
        return bool(re.search(r"[\u4e00-\u9fff]", text_clean)) and len(text_clean) >= 4

    elif language.lower() == "english":
        if re.search(r"[\u4e00-\u9fff\u0900-\u097f]", text_clean):
            return False
        words = [w for w in text_clean.split() if not w.startswith("@") and not w.startswith("http")]
        if len(words) < 4 or text_clean.lower() in PINYIN_SLANG:
            return False
        return bool(re.search(r"[a-zA-Z]{3,}", text_clean))

    return False


def infer_habits(comments: list[str], language: str) -> dict[str, Any]:
    if language.lower() == "chinese":
        total_chars = sum(len(c) for c in comments)
        avg_len = total_chars / max(len(comments), 1)
        punct = sum(bool(re.search(r"[！？!?]{1,}|(\.\.|\.\.\.|……)", c)) for c in comments) / len(comments)
        emoji = sum(bool(re.search(r"[😀-🙏]|\[[a-zA-Z\u4e00-\u9fff]+\]", c)) for c in comments) / len(comments)

        traits = ["简短精炼" if avg_len < 20 else "表达详实"]
        if punct > 0.25:
            traits.append("语气强烈")
        if emoji > 0.20:
            traits.append("喜用表情包")

        return {
            "summary": "；".join(traits) + "。",
            "formality": "口语化" if avg_len < 20 else "书面表达",
            "avg_unit_length": round(avg_len, 1),
            "unit": "characters",
            "uses_emoji": emoji > 0.20,
            "emphatic_punctuation": punct > 0.25,
        }
    else:
        total_words = sum(len(c.split()) for c in comments)
        avg_words = total_words / max(len(comments), 1)
        punct = sum(bool(re.search(r"[!?]{2,}|\.{3,}", c)) for c in comments) / len(comments)
        emoji = sum(bool(re.search(r"[😀-🙏]|[\U0001F600-\U0001F64F]", c)) for c in comments) / len(comments)

        traits = ["usually concise" if avg_words < 12 else "expressive and detailed"]
        if punct > 0.25:
            traits.append("uses emphatic punctuation")
        if emoji > 0.20:
            traits.append("regularly uses emojis")

        return {
            "summary": "; ".join(traits) + ".",
            "formality": "Colloquial" if avg_words < 12 else "Formal",
            "avg_unit_length": round(avg_words, 1),
            "unit": "words",
            "uses_emoji": emoji > 0.20,
            "emphatic_punctuation": punct > 0.25,
        }


def map_archetype(idx: int, profile_text: str, language: str) -> str:
    p_lower = profile_text.lower()
    if language.lower() == "chinese":
        if any(w in p_lower for w in ["平权", "包容", "少数群体", "彩虹", "自由", "ally"]):
            return "Supportive / 包容友善支持者"
        if any(w in p_lower for w in ["传统", "礼序", "家庭", "保守", "国学"]):
            return "Cultural / 传统价值观持有者"
        if any(w in p_lower for w in ["科普", "学者", "教师", "心理", "法学"]):
            return "Educator / 事实科普与理性分析者"

        archetypes = [
            "Supportive / 包容友善支持者",
            "Critical / 批判保守者",
            "Cultural / 传统价值观持有者",
            "Skeptical / 怀疑调侃者",
            "Educator / 事实科普与理性分析者",
            "Neutral Observer / 客观旁观者",
        ]
        return archetypes[idx % len(archetypes)]
    else:
        if any(w in p_lower for w in ["ally", "equality", "rights", "liberal"]):
            return "Supportive / Pro-LGBTQ+ Ally"
        if any(w in p_lower for w in ["tradition", "faith", "conservative"]):
            return "Cultural / Conservative Norms"
        if any(w in p_lower for w in ["phd", "researcher", "facts", "analyst"]):
            return "Educator / Factual Explainer"

        archetypes = [
            "Supportive / Pro-LGBTQ+ Ally",
            "Critical / Conservative",
            "Cultural / Traditional Norms",
            "Skeptical / Ambiguous Stance",
            "Educator / Factual Explainer",
            "Neutral Observer",
        ]
        return archetypes[idx % len(archetypes)]


def build_persona_pool(raw_data: Any, language: str = "Chinese", max_pool_size: int = 300) -> list[dict]:
    user_records = defaultdict(lambda: {"comments": [], "profiles": []})

    def first_match(record: dict, key_set: set[str]) -> str:
        for k, v in record.items():
            if re.sub(r"[^a-z0-9]", "", str(k).lower()) in key_set and isinstance(v, (str, int, float)):
                c = str(v).strip()
                if c:
                    return c
        return ""

    def ingest(uid: str, rec: dict):
        comment = first_match(rec, COMMENT_KEYS)
        if not comment or not is_valid_comment(comment, language):
            return
        prof = first_match(rec, PROFILE_KEYS)
        user_records[str(uid)]["comments"].append(comment)
        if prof:
            user_records[str(uid)]["profiles"].append(prof)

    if isinstance(raw_data, dict):
        for uid, val in raw_data.items():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        ingest(uid, item)
            elif isinstance(val, dict):
                ingest(uid, val)
    elif isinstance(raw_data, list):
        for idx, item in enumerate(raw_data):
            if isinstance(item, dict):
                uid = first_match(item, USER_KEYS) or f"user_{idx}"
                ingest(uid, item)

    prefix = "P_ZH" if language.lower() == "chinese" else "P_EN"
    default_bio = "社交网络普通活跃用户。" if language.lower() == "chinese" else "Active social media user."

    pool = []
    for count, (uid, data) in enumerate(user_records.items(), start=1):
        if not data["comments"]:
            continue

        anon_hash = hashlib.sha256(str(uid).encode("utf-8")).hexdigest()[:12]
        profile_str = " ".join(dict.fromkeys(data["profiles"]))[:300]
        habits = infer_habits(data["comments"], language)

        pool.append({
            "persona_id": f"{prefix}_{count:04d}",
            "source_user_hash": anon_hash,
            "language": language.capitalize(),
            "archetype": map_archetype(count, profile_str, language),
            "profile_bio": profile_str or default_bio,
            "commenting_habits": habits,
            "sample_historical_comments": data["comments"][:6],
            "total_available_comments": len(data["comments"]),
            "provenance": "Mined from SoMe social benchmark."
        })

        if len(pool) >= max_pool_size:
            break

    return pool








































# """Module for extracting, anonymizing, and structuring social media personas from SoMe/Twitter data."""

# from __future__ import annotations

# import hashlib
# import json
# import re
# from collections import Counter, defaultdict
# from pathlib import Path
# from typing import Any
# import requests

# USER_KEYS = {"userid", "user", "username", "screenname", "author", "authorid", "accountid", "handle"}
# COMMENT_KEYS = {"comment", "text", "content", "tweet", "post", "reply", "commenttext", "usercomment"}
# PROFILE_KEYS = {"profile", "userprofile", "description", "bio", "userdescription", "occupation"}
# STANCE_KEYS = {"politicalstance", "stance", "ideology", "attitude"}


# def is_pure_english(text: str) -> bool:
#     """Ensures text uses Latin script and excludes Chinese Hanzi / Hindi Devanagari."""
#     if re.search(r"[\u4e00-\u9fff]", text) or re.search(r"[\u0900-\u097f]", text):
#         return False
#     return bool(re.search(r"[a-zA-Z]", text))


# def infer_commenting_habits(comments: list[str]) -> dict[str, Any]:
#     """Computes observable behavioral traits without making demographic assumptions."""
#     if not comments:
#         return {"summary": "Neutral and concise.", "avg_words": 0, "emoji": False, "punct": False}
#     avg_words = sum(len(c.split()) for c in comments) / len(comments)
#     punct_ratio = sum(bool(re.search(r"[!?]{2,}|\.{3,}", c)) for c in comments) / len(comments)
#     emoji_ratio = sum(bool(re.search(r"[😀-🙏]|[\U0001F600-\U0001F64F]", c)) for c in comments) / len(comments)

#     traits = ["usually concise" if avg_words < 12 else "expressive and detailed"]
#     if punct_ratio > 0.25:
#         traits.append("uses emphatic punctuation")
#     if emoji_ratio > 0.20:
#         traits.append("regularly uses emojis")

#     return {
#         "summary": "; ".join(traits) + ".",
#         "avg_words": round(avg_words, 1),
#         "uses_emoji": emoji_ratio > 0.20,
#         "emphatic_punctuation": punct_ratio > 0.25,
#     }


# def map_archetype(idx: int, profile_text: str) -> str:
#     """Assigns diverse perspectives matching your framework diagram."""
#     p_lower = profile_text.lower()
#     if any(w in p_lower for w in ["ally", "equality", "rights", "liberal"]):
#         return "Supportive / Pro-LGBTQ+ Ally"
#     if any(w in p_lower for w in ["tradition", "faith", "conservative"]):
#         return "Cultural / Conservative Norms"
#     if any(w in p_lower for w in ["phd", "researcher", "facts", "analyst"]):
#         return "Educator / Factual Explainer"

#     archetypes = [
#         "Supportive / Pro-LGBTQ+ Ally",
#         "Critical / Conservative",
#         "Cultural / Traditional Norms",
#         "Skeptical / Ambiguous Stance",
#         "Educator / Factual Explainer",
#         "Neutral Observer",
#     ]
#     return archetypes[idx % len(archetypes)]


# def build_persona_pool(raw_data: Any, language: str = "English", max_pool_size: int = 300) -> list[dict]:
#     """Ingests raw social records, filters by target language, and compiles standardized personas."""
#     user_records = defaultdict(lambda: {"comments": [], "profiles": []})

#     def first_match(record: dict, key_set: set[str]) -> str:
#         for k, v in record.items():
#             if re.sub(r"[^a-z0-9]", "", str(k).lower()) in key_set and isinstance(v, (str, int, float)):
#                 c = str(v).strip()
#                 if c:
#                     return c
#         return ""

#     def ingest(uid: str, rec: dict):
#         comment = first_match(rec, COMMENT_KEYS)
#         if not comment or len(comment) < 3:
#             return
#         if language == "English" and not is_pure_english(comment):
#             return
#         prof = first_match(rec, PROFILE_KEYS)
#         user_records[str(uid)]["comments"].append(comment)
#         if prof and (language != "English" or is_pure_english(prof)):
#             user_records[str(uid)]["profiles"].append(prof)

#     if isinstance(raw_data, dict):
#         for uid, val in raw_data.items():
#             if isinstance(val, list):
#                 for item in val:
#                     if isinstance(item, dict):
#                         ingest(uid, item)
#             elif isinstance(val, dict):
#                 ingest(uid, val)
#     elif isinstance(raw_data, list):
#         for idx, item in enumerate(raw_data):
#             if isinstance(item, dict):
#                 uid = first_match(item, USER_KEYS) or f"user_{idx}"
#                 ingest(uid, item)

#     pool = []
#     for count, (uid, data) in enumerate(user_records.items(), start=1):
#         if not data["comments"]:
#             continue
#         anon_hash = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12]
#         profile_str = " ".join(dict.fromkeys(data["profiles"]))[:300]
#         habits = infer_commenting_habits(data["comments"])

#         pool.append({
#             "persona_id": f"P_EN_{count:04d}",
#             "source_user_hash": anon_hash,
#             "language": language,
#             "archetype": map_archetype(count, profile_str),
#             "profile_bio": profile_str or "Active social media user.",
#             "commenting_habits": habits,
#             "sample_historical_comments": data["comments"][:6],
#             "total_available_comments": len(data["comments"]),
#         })

#         if len(pool) >= max_pool_size:
#             break

#     return pool