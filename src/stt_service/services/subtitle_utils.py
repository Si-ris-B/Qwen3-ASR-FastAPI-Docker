import numpy as np

def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def join_tokens(a: str, b: str) -> str:
    if not a: return b
    if not b: return a
    # ComfyUI Logic: Don't add spaces if characters are CJK
    for ch in (a[-1], b[0]):
        if "\u4e00" <= ch <= "\u9fff":
            return f"{a}{b}"
    return f"{a} {b}"


def group_time_stamps(time_stamps, max_gap_sec: float, max_chars: int, split_mode: str):
    if not time_stamps: return []
    groups = []
    cur = None
    punct = ("。", "！", "？", ".", "!", "?")

    for item in time_stamps:
        text = (item.text or "").strip()
        if not text: continue
        if cur is None:
            cur = {"start": item.start_time, "end": item.end_time, "text": text}
            continue

        gap = float(item.start_time) - float(cur["end"])
        too_far = gap > max_gap_sec
        too_long = max_chars > 0 and (len(cur["text"]) + len(text)) > max_chars
        end_sentence = any(cur["text"].endswith(p) for p in punct)

        should_split = False
        if "punctuation" in split_mode and end_sentence: should_split = True
        if "length" in split_mode and too_long: should_split = True
        if "pause" in split_mode and too_far: should_split = True

        if should_split:
            groups.append(cur)
            cur = {"start": item.start_time, "end": item.end_time, "text": text}
        else:
            cur["text"] = join_tokens(cur["text"], text).strip()
            cur["end"] = item.end_time

    if cur is not None: groups.append(cur)
    return groups