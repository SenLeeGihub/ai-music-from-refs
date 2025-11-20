"""
Generate Chinese lyrics based on style summaries using the OpenAI Responses API.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

from openai import OpenAI, OpenAIError


def _create_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required to generate lyrics.")
    return OpenAI(api_key=api_key)


def _format_style_context(style_profile: Dict) -> str:
    tempo_range = style_profile.get("tempo_range")
    tempo_text = ""
    if tempo_range and len(tempo_range) == 2:
        tempo_text = f"节奏范围约 {tempo_range[0]:.0f}-{tempo_range[1]:.0f} BPM。"

    energy = style_profile.get("energy_profile", {})
    if energy:
        ordered_bands = []
        for band in ("low", "mid", "high"):
            if band in energy:
                ordered_bands.append(f"{band}={energy[band]:.2f}")
        energy_text = "频段能量：" + "，".join(ordered_bands)
    else:
        energy_text = "频段能量分布均衡。"
    return f"{tempo_text} {energy_text}".strip()


def _build_messages(style_profile: Dict, style_text: str, theme: Optional[str]) -> List[Dict[str, str]]:
    style_context = _format_style_context(style_profile)
    topic_text = theme or "由你推断的主题"
    user_prompt = (
        "你是一名专业华语作词人。根据以下音乐风格信息与主题，"
        "创作完整中文歌词，至少包含两个 verse 和一个 chorus，可选 pre-chorus。"
        "歌词需押韵、富有画面感，正文必须是中文，标题可以使用简短英文。"
        "整首歌控制在 3-5 个段落，每段 4-8 行。最终使用 JSON 输出，字段结构严格如下：\n"
        '{\n'
        '  "title": "歌名",\n'
        '  "language": "zh",\n'
        '  "theme": "主题",\n'
        '  "sections": [\n'
        '    {"type": "verse", "name": "Verse 1", "lines": ["...", "..."]},\n'
        '    {"type": "pre_chorus", "name": "Pre-Chorus", "lines": ["...", "..."]},\n'
        '    {"type": "chorus", "name": "Chorus", "lines": ["...", "..."]}\n'
        '  ]\n'
        '}\n'
        "所有段落行请使用字符串数组表示，不要添加额外说明。\n\n"
        f"音乐风格总结：{style_text.strip()}\n"
        f"数值特征：{style_context}\n"
        f"主题：{topic_text}\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You write poetic Mandarin lyrics tailored for AI music generation projects. "
                "Always respond in Chinese unless explicitly asked otherwise."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _extract_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    if output:
        chunks = []
        for item in output:
            for content in getattr(item, "content", []):
                content_text = getattr(content, "text", None)
                if content_text:
                    chunks.append(content_text)
        if chunks:
            return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    raise RuntimeError("Failed to extract lyrics text from OpenAI response")


def _normalize_sections(sections: Optional[List[Dict]]) -> List[Dict]:
    normalized = []
    if not isinstance(sections, list):
        return normalized
    for idx, entry in enumerate(sections, start=1):
        if not isinstance(entry, dict):
            continue
        section_type = entry.get("type") or "verse"
        name = entry.get("name") or f"Section {idx}"
        lines = entry.get("lines")
        if isinstance(lines, list):
            clean_lines = [str(line).strip() for line in lines if str(line).strip()]
        elif isinstance(lines, str):
            clean_lines = [line.strip() for line in lines.splitlines() if line.strip()]
        else:
            clean_lines = []
        if not clean_lines:
            continue
        normalized.append(
            {
                "type": section_type,
                "name": name,
                "lines": clean_lines,
            }
        )
    return normalized


def _text_fallback_to_structure(text: str) -> Dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = ["无歌词内容"]
    return {
        "title": "未命名作品",
        "language": "zh",
        "theme": None,
        "sections": [
            {
                "type": "verse",
                "name": "Verse 1",
                "lines": lines,
            }
        ],
    }


def _parse_json_safe(text: str) -> Dict:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data.setdefault("language", "zh")
            sections = _normalize_sections(data.get("sections"))
            if not sections and data.get("lyrics"):
                fallback = _text_fallback_to_structure(data["lyrics"])
                fallback["title"] = data.get("title") or fallback["title"]
                fallback["theme"] = data.get("theme")
                return fallback
            if sections:
                data["sections"] = sections
                return data
    except json.JSONDecodeError:
        pass
    return _text_fallback_to_structure(text)


def generate_lyrics(style_profile: Dict, style_text: str, theme: str | None = None) -> Dict:
    """
    使用 OpenAI Responses API，根据 style_profile + style_text 自动生成歌曲歌词（中文）。
    Returns a dictionary with at least title/theme/lyrics keys.
    """
    client = _create_openai_client()
    messages = _build_messages(style_profile, style_text, theme)
    try:
        response = client.responses.create(model="gpt-5.1", input=messages)
    except OpenAIError as exc:
        raise RuntimeError("Failed to generate lyrics") from exc

    text = _extract_response_text(response)
    result = _parse_json_safe(text)
    if theme and not result.get("theme"):
        result["theme"] = theme
    result.setdefault("language", "zh")
    return result


def lyrics_to_text(lyrics: Dict) -> str:
    """
    把结构化歌词字典转成可读多行文本，便于保存为 .txt。
    """
    title = str(lyrics.get("title") or "未命名作品").strip()
    theme = lyrics.get("theme")
    language = lyrics.get("language")
    sections = lyrics.get("sections") or []

    lines: List[str] = [title]
    meta_parts: List[str] = []
    if theme:
        meta_parts.append(f"主题：{theme}")
    if language:
        meta_parts.append(f"语言：{language}")
    if meta_parts:
        lines.append(" / ".join(meta_parts))
        lines.append("")

    if isinstance(sections, list) and sections:
        for idx, section in enumerate(sections, start=1):
            if not isinstance(section, dict):
                continue
            sec_name = str(section.get("name") or f"Section {idx}")
            sec_type = section.get("type")
            header = sec_name
            if sec_type and sec_type.lower() not in sec_name.lower():
                header = f"{sec_name} ({sec_type})"
            lines.append(f"[{header}]")
            sec_lines = section.get("lines")
            if isinstance(sec_lines, str):
                candidate_lines = [sec_lines]
            else:
                candidate_lines = sec_lines or []
            for line in candidate_lines:
                line_text = str(line).strip()
                if line_text:
                    lines.append(line_text)
            lines.append("")
    else:
        fallback = lyrics.get("lyrics")
        if isinstance(fallback, str) and fallback.strip():
            lines.append(fallback.strip())

    return "\n".join(lines).strip()
