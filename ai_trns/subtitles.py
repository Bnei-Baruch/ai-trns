"""Rendering a words document (ai_trns.words) to text / SRT / JSON"""
import json
import re

from . import cues

FORMATS = ("txt", "srt", "json")


def render(doc: dict, fmt: str, fix_rtl: bool = False) -> str:
    """Words document -> txt | srt | json. CPU only, so any format can be rendered again later.
    SRT uses the whisper_trainer cue layout (ai_trns.cues), or is the SRT the document came with
    (ai_trns.stt_service); fix_rtl adds RTL marks to Hebrew"""
    if fmt == "txt":
        return doc["text"]
    if fmt == "json":
        return json.dumps(doc, ensure_ascii=False)
    if fmt != "srt":
        raise ValueError(f"unknown format {fmt!r}")
    content = doc["srt"] if "srt" in doc else cues.render_srt(doc)
    return fix_rtl_srt(content) if fix_rtl and doc.get("language") == "he" else content


def fix_rtl_srt(content: str) -> str:
    """RTL marks on the text lines of an SRT (index and timing lines are kept)"""
    out = []
    for block in content.split("\n\n"):
        lines = block.split("\n")
        out.append("\n".join(lines[:2] + [fix_rtl_subtitle_line(line) for line in lines[2:]]))
    return "\n\n".join(out)


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
