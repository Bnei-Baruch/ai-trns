"""Rendering a words document (ai_trns.words) to text / SRT / JSON"""
import json
import os
import re
from datetime import timedelta
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from typing import Optional

from . import cues

FORMATS = ("txt", "srt", "json")
# house: whisper_trainer cue policy (ai_trns.cues); advanced: ToSubs; standard: split segments by duration
SRT_STYLES = ("house", "advanced", "standard")

# Adding new imports for advanced SRT generation
try:
    import pysrt
    from pysrt import SubRipItem
    ADVANCED_SRT_AVAILABLE = True
except ImportError:
    ADVANCED_SRT_AVAILABLE = False
    print("[!] To use advanced SRT generation, install: pip install pysrt")


def render(doc: dict, fmt: str, srt_style: str = "house", fix_rtl: bool = False) -> str:
    """Words document -> txt | srt | json. CPU only, so any format can be rendered again later"""
    if fmt == "txt":
        return doc["text"]
    if fmt == "json":
        return json.dumps(doc, ensure_ascii=False)
    if fmt != "srt":
        raise ValueError(f"unknown format {fmt!r}")
    if srt_style == "house":
        content = cues.render_srt(doc)
        return fix_rtl_srt(content) if fix_rtl else content
    segments = _segments_of(doc)
    lang = doc.get("language") or "auto"
    if srt_style == "advanced" and ADVANCED_SRT_AVAILABLE:
        content = segments_to_advanced_srt(segments, lang, fix_rtl)
        if content is not None:
            return content
        print("[!] Could not get word timestamps, using standard generation")
    return format_segments_to_srt(segments)


def _segments_of(doc: dict) -> list:
    """Segment objects in faster-whisper's shape, for the legacy renderers below"""
    segments = [SimpleNamespace(start=s["start_s"], end=s["end_s"], text=s["text"], words=[])
                for s in doc["segments"]]
    for w in doc["words"]:
        segments[w["segment"]].words.append(SimpleNamespace(start=w["start_s"], end=w["end_s"], word=w["word"]))
    return segments


def fix_rtl_srt(content: str) -> str:
    """RTL marks on the text lines of an SRT (index and timing lines are kept)"""
    out = []
    for block in content.split("\n\n"):
        lines = block.split("\n")
        out.append("\n".join(lines[:2] + [fix_rtl_subtitle_line(line) for line in lines[2:]]))
    return "\n\n".join(out)


def split_segment_text_by_words(text, parts, min_words_per_chunk=2):
    words = text.strip().split()
    total_words = len(words)
    base = total_words // parts
    extra = total_words % parts
    out = []
    last = 0
    # Split as before
    for i in range(parts):
        take = base + (1 if i < extra else 0)
        out.append(words[last:last+take])
        last += take
    # Merge chunks that are too short with previous one
    i = 0
    while i < len(out):
        if len(out[i]) < min_words_per_chunk:
            if i > 0:
                out[i-1].extend(out[i])
                out.pop(i)
            elif i+1 < len(out):
                out[i+1] = out[i] + out[i+1]
                out.pop(i)
            else:
                i += 1
        else:
            i += 1
    # Rebuild strings
    return [" ".join(chunk).strip() for chunk in out if chunk]

def format_ts(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def format_segments_to_srt(segments, max_segment_length=10.0):
    lines = []
    idx = 1
    for segment in segments:
        seg_start = segment.start
        seg_end = segment.end
        seg_text = segment.text.strip()
        duration = seg_end - seg_start

        num_chunks = max(1, int((duration + max_segment_length - 1) // max_segment_length))
        if num_chunks == 1:
            lines.append(f"{idx}")
            lines.append(f"{format_ts(seg_start)} --> {format_ts(seg_end)}")
            lines.append(seg_text)
            lines.append("")
            idx += 1
        else:
            text_chunks = split_segment_text_by_words(seg_text, num_chunks)
            chunk_duration = duration / len(text_chunks)
            for i in range(len(text_chunks)):
                chunk_s = seg_start + i * chunk_duration
                chunk_e = min(seg_end, chunk_s + chunk_duration)
                if chunk_s >= chunk_e or not text_chunks[i].strip():
                    continue
                lines.append(f"{idx}")
                lines.append(f"{format_ts(chunk_s)} --> {format_ts(chunk_e)}")
                lines.append(text_chunks[i])
                lines.append("")
                idx += 1
    return "\n".join(lines)


# Constants for advanced SRT generation
MAX_SUB_LENGTH = 60
MIN_SUB_LENGHT = 10
MAX_LINE_LENGTH = MAX_SUB_LENGTH / 2

sent_pattern = r"[!.?]"
pattern = r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"""
dash_pattern = r"(^|\s)-"

# Unicode symbols for RTL text direction control
RLE = '\u202B'  # Right-to-Left Embedding
PDF = '\u202C'  # Pop Directional Formatting  
RLM = '\u200F'  # Right-to-Left Mark
LRM = '\u200E'  # Left-to-Right Mark

def fix_rtl_punctuation(text: str, is_hebrew: bool = True) -> str:
    """
    Fixes punctuation display issues in RTL text
    """
    if not is_hebrew:
        return text
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text.strip())
    
    # List of punctuation marks to fix
    punctuation_marks = r'[.!?,:;]'
    
    # Add RLM after punctuation for correct display
    text = re.sub(f'({punctuation_marks})(\\s+)', r'\1' + RLM + r'\2', text)
    
    # Fix number issues - add LRM around English numbers
    text = re.sub(r'(\d+)', LRM + r'\1' + LRM, text)
    
    # Fix English words in Hebrew text
    english_pattern = r'([a-zA-Z]+)'
    text = re.sub(english_pattern, LRM + r'\1' + LRM, text)
    
    # Wrap entire text in RLE...PDF for forced RTL
    text = RLE + text + PDF
    
    # Remove duplicate control characters
    text = re.sub(f'{RLM}+', RLM, text)
    text = re.sub(f'{LRM}+', LRM, text)
    
    return text

def fix_rtl_subtitle_line(line: str, is_hebrew: bool = True) -> str:
    """
    Fixes a single subtitle line for RTL
    """
    if not is_hebrew or not line.strip():
        return line
        
    # Split on newlines if present
    lines = line.split('\n')
    fixed_lines = []
    
    for single_line in lines:
        if single_line.strip():
            fixed_line = fix_rtl_punctuation(single_line.strip(), is_hebrew)
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(single_line)
    
    return '\n'.join(fixed_lines)


class ToSubs:
    """Advanced SRT subtitle generator from the second project"""
    def __init__(self, times, lang, fix_rtl=False):
        self.times = times
        self.last_idx = len(times) - 1
        # For auto language detection, we need to detect Hebrew text
        if lang == "auto":
            # Try to detect Hebrew in the text
            sample_text = ""
            for i in range(min(10, len(times))):  # Sample first 10 words
                if i in times:
                    sample_text += times[i].get('word', '') + " "
            self.is_heb = self._detect_hebrew(sample_text)
        else:
            self.is_heb = lang == "he"
        self.fix_rtl = fix_rtl and self.is_heb  # Apply RTL only for Hebrew
    
    def _detect_hebrew(self, text: str) -> bool:
        """Simple Hebrew detection based on Unicode ranges"""
        if not text:
            return False
        hebrew_chars = 0
        latin_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                # Hebrew Unicode range: U+0590 to U+05FF
                if '\u0590' <= char <= '\u05FF':
                    hebrew_chars += 1
                # Latin characters (English)
                elif 'a' <= char.lower() <= 'z':
                    latin_chars += 1
        
        if total_chars == 0:
            return False
        
        # If Hebrew chars are more than 50% or 
        # Hebrew chars are more than Latin chars, consider it Hebrew text
        hebrew_ratio = hebrew_chars / total_chars
        return hebrew_ratio > 0.5 or (hebrew_chars > latin_chars and hebrew_ratio > 0.2)

    def run(self):
        if not ADVANCED_SRT_AVAILABLE:
            raise ImportError("pysrt not installed. Use: pip install pysrt")
        
        res_file = pysrt.SubRipFile()
        idx = 0
        sub_idx = 0
        while not idx == self.last_idx:
            (n_idx, prior) = self.find_next_idx_by_symbol(idx)
            if prior < 2:
                (n_idx, pause) = self.find_next_idx_by_pause(idx)
                if pause == 0:
                    n_idx = self.find_next_idx_by_len(idx)
            if idx == n_idx:
                n_idx += 1
            n_idx = min(n_idx, self.last_idx)

            txt = self.get_sub_text(idx, n_idx)
            start = self.sub_by_idx(idx)['start']
            end = self.patch_end_time(n_idx)
            sub = SubRipItem(sub_idx, pysrt.SubRipTime(seconds=start), pysrt.SubRipTime(seconds=end), txt)
            res_file.append(sub)

            idx = n_idx
            sub_idx += 1
        return res_file

    def find_next_idx_by_symbol(self, start_idx):
        ln = 0
        dash_count = 1
        sent_count = 0
        punct_count = 0
        max_prior = (0, 0)
        idx = start_idx
        while ln < MAX_SUB_LENGTH and self.has_idx(idx):
            w = self.sub_by_idx(idx)['word']
            ln += len(w)
            idx += 1

            if self.is_heb and re.search(dash_pattern, w) is not None and start_idx != idx - 1:
                dash_count += 1
                next_prior = dash_count * 9
                if next_prior > max_prior[1]:
                    max_prior = (idx - 1, dash_count * 9)
            if dash_count == 3:
                max_prior = (idx - 1, 100)
                break

            if ln < MIN_SUB_LENGHT:
                continue
            if self.is_heb and re.search(dash_pattern, w) is not None and ln > MAX_LINE_LENGTH:
                max_prior = (idx - 1, 100)
                break
            if re.search(sent_pattern, w) is not None:
                sent_count += 1
                max_prior = (idx, sent_count * 10)
                continue
            if max_prior[1] < 10 and re.search(pattern, w) is not None:
                punct_count += 1
                max_prior = (idx, punct_count * 2)
                continue
        return max_prior

    def find_next_idx_by_len(self, idx):
        ln = 0
        while ln < MAX_SUB_LENGTH and self.has_idx(idx):
            w = self.sub_by_idx(idx)['word']
            ln += len(w)
            idx += 1
        return idx

    def find_next_idx_by_pause(self, idx):
        ln = 0
        max_pause_idx = (idx, 0)
        while ln < MAX_SUB_LENGTH and self.has_idx(idx):
            w = self.sub_by_idx(idx)['word']
            ln += len(w)
            idx += 1
            if ln < MIN_SUB_LENGHT:
                continue

            pause = self.calc_pause(idx)
            if pause >= max_pause_idx[1]:
                max_pause_idx = (idx, pause)

            if ln > MAX_SUB_LENGTH:
                break
        return max_pause_idx

    def calc_pause(self, idx):
        # for last item
        if not self.has_idx(idx + 1):
            return 1000
        return self.sub_by_idx(idx + 1)['start'] - self.sub_by_idx(idx)['end']

    def get_sub_text(self, start_idx, end_idx):
        data = []
        offset = 0
        has_dash = False
        for idx in range(start_idx, end_idx):
            w = self.sub_by_idx(idx)['word'].strip()
            _len = len(w)
            if self.is_heb and re.search(dash_pattern, w) and idx != start_idx:
                w = f"\n{w}"
                has_dash = True
            data.append(({"w": w, "o": offset, "l": _len}))
            offset += _len
        
        if offset < MAX_LINE_LENGTH or has_dash:
            result = " ".join(list(map(lambda d: d["w"], data)))
        else:
            median = offset / 2
            idx = 0
            diff = 0.5
            for i in range(len(data) - 1):
                if re.search(pattern, data[i]["w"]):
                    _diff = abs(data[i]["o"] + data[i]["l"] - median) / median
                    if _diff < diff:
                        diff = _diff
                        idx = i
            if idx == 0:
                for i in range(len(data) - 1):
                    if data[i]["o"] + data[i]["l"] > median >= data[i]["o"] - 1:
                        idx = i - 1

            data[idx]["w"] = f"{data[idx]['w']}\n"
            result = " ".join(list(map(lambda d: f"{d['w']}", data)))
        
        # Apply RTL fixes for Hebrew
        if self.fix_rtl:
            result = fix_rtl_subtitle_line(result, True)
        
        return result

    def sub_by_idx(self, idx):
        return self.times[idx]

    def has_idx(self, idx):
        return idx in self.times

    def patch_end_time(self, idx):
        end = self.sub_by_idx(idx - 1)['end']
        if not self.has_idx(idx + 1):
            return end
        n_start = self.sub_by_idx(idx)['start']
        return n_start - max(0.1, (n_start - end) / 2)


def segments_to_advanced_srt(segments, lang: str, fix_rtl: bool) -> Optional[str]:
    """Build advanced SRT from segments with words; None if there are no word timestamps"""
    # Convert segments to format expected by ToSubs
    word_timestamps = {}
    idx = 0
    for segment in segments:
        if hasattr(segment, 'words') and segment.words:
            for word in segment.words:
                word_timestamps[idx] = {
                    'start': word.start,
                    'end': word.end,
                    'word': word.word
                }
                idx += 1
    
    if not word_timestamps:
        return None
    
    # Generate SRT through ToSubs with RTL fixes
    to_subs = ToSubs(word_timestamps, lang, fix_rtl=fix_rtl)
    srt_file = to_subs.run()
    
    # Use built-in pysrt method for string conversion
    # This is more reliable than join
    
    with NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as temp_file:
        srt_file.save(temp_file.name, encoding='utf-8')
    
    # Read content and delete temporary file
    try:
        with open(temp_file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        os.unlink(temp_file.name)
        return content
    except Exception as e:
        print(f"[!] Error reading SRT file: {e}")
        # Fallback to previous method
        return '\n'.join(str(item) for item in srt_file)
