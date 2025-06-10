from faster_whisper import WhisperModel
from tempfile import NamedTemporaryFile
import wave
import numpy as np
from typing import Iterator
from datetime import timedelta
import re

# Adding new imports for advanced SRT generation
try:
    import pysrt
    from pysrt import SubRipItem
    ADVANCED_SRT_AVAILABLE = True
except ImportError:
    ADVANCED_SRT_AVAILABLE = False
    print("[!] To use advanced SRT generation, install: pip install pysrt")

model = None  # Will be initialized via load_model()

PROFILES = {
    "fast": dict(
        language="he",
        vad_filter=True,
        beam_size=3,
        temperature=[0.0],
        no_speech_threshold=0.5
    ),
    "stable": dict(
        language="he",
        vad_filter=False,
        beam_size=8,
        temperature=[0.0, 0.2],
        no_speech_threshold=0.7,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4
    )
}

transcribe_kwargs = PROFILES["stable"]

def load_model(lang: str):
    if lang == "auto":
        # For auto language detection, use English model as base multilingual model
        model_path = "./models/en"
    else:
        model_path = f"./models/{lang}"
    
    global model
    model = WhisperModel(
        model_path,
        device="cuda",
        compute_type="float16"
    )
    
    if lang == "auto":
        # Set language to None for auto-detection
        lang_code = None
    else:
        # he_bb should use "he" as language code but different model path
        lang_code = lang if lang in ["en", "ru", "he"] else "he"
    
    for profile in PROFILES.values():
        profile["language"] = lang_code

def transcribe_file(filename: str) -> str:
    segments, _ = model.transcribe(filename, **transcribe_kwargs)
    return " ".join([s.text.strip() for s in segments])

def transcribe_file_to_srt(filename: str, max_segment_length: float = 10.0) -> str:
    segments, _ = model.transcribe(filename, **transcribe_kwargs)
    return format_segments_to_srt(segments, max_segment_length=max_segment_length)

def transcribe_array_to_text(audio_array: np.ndarray) -> str:
    segments, _ = model.transcribe(audio_array, **transcribe_kwargs)
    return " ".join([s.text.strip() for s in segments])

def transcribe_array_to_srt(audio_array: np.ndarray, max_segment_length: float = 10.0) -> str:
    segments, _ = model.transcribe(audio_array, **transcribe_kwargs)
    return format_segments_to_srt(segments, max_segment_length=max_segment_length)

def transcribe_stream(byte_stream: Iterator[bytes],
                      chunk_duration_sec: float = 6.0,
                      sample_rate: int = 16000,
                      bytes_per_sample: int = 2) -> Iterator[str]:
    buffer = b""
    threshold = int(chunk_duration_sec * sample_rate * bytes_per_sample)

    for chunk in byte_stream:
        buffer += chunk
        if len(buffer) >= threshold:
            with NamedTemporaryFile(suffix=".wav", delete=True) as f:
                with wave.open(f.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(bytes_per_sample)
                    wf.setframerate(sample_rate)
                    wf.writeframes(buffer)

                buffer = b""

                segments, _ = model.transcribe(f.name, **transcribe_kwargs)
                yield " ".join([s.text.strip() for s in segments])

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
            self.is_heb = lang in ["he", "he_bb"]
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


def transcribe_file_to_advanced_srt(filename: str, lang: str = "he", fix_rtl: bool = False) -> str:
    """
    Advanced SRT generation with pause, punctuation and context analysis
    Uses logic from the second mdb-ai project
    
    Args:
        filename: path to audio file
        lang: transcription language
        fix_rtl: fix RTL punctuation issues (for Hebrew)
    """
    if not ADVANCED_SRT_AVAILABLE:
        print("[!] pysrt not available, using standard SRT generation")
        return transcribe_file_to_srt(filename)
    
    # Use transcription settings with word_timestamps
    kwargs = transcribe_kwargs.copy()
    kwargs['word_timestamps'] = True
    
    segments, _ = model.transcribe(filename, **kwargs)
    
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
        print("[!] Could not get word timestamps, using standard generation")
        return transcribe_file_to_srt(filename)
    
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
        import os
        os.unlink(temp_file.name)
        return content
    except Exception as e:
        print(f"[!] Error reading SRT file: {e}")
        # Fallback to previous method
        return '\n'.join(str(item) for item in srt_file)

