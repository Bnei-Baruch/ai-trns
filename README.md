# Audio Transcription with Whisper

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Transcription
```bash
python whisper_fullfile_cli.py --input audio.mp3 --txt --lang ru
```

### Subtitle Generation

#### Standard SRT Subtitles (default):
```bash
python whisper_fullfile_cli.py --input audio.mp3 --srt --lang he
# or explicitly specify implementation 1:
python whisper_fullfile_cli.py --input audio.mp3 --srt --impl 1 --lang he
```

#### 🆕 **Advanced SRT Subtitles** (NEW!):
```bash
python whisper_fullfile_cli.py --input audio.mp3 --srt --impl 2 --lang he
```

#### 🔥 **Advanced SRT with RTL Fixes** (SUPER NEW!):
```bash
python whisper_fullfile_cli.py --input audio.mp3 --srt --impl 2 --lang he --fix-rtl
```

## New Feature: SRT Implementation Selection

### `--impl` Parameter:

**`--impl 1`** (default) - **Standard Implementation:**
- Splitting by segment duration
- Fast generation
- Universal for all languages

**`--impl 2`** - **Advanced Implementation** from mdb-ai project:
- 🎯 **Pause analysis** between words
- 📝 **Intelligent splitting** by punctuation (!.?)
- 🔍 **Special Hebrew text handling** with dashes
- 📏 **Optimal line length** (up to 60 characters)
- ⏱️ **Precise timestamp synchronization**

### 🆕 **`--fix-rtl` Parameter** (only for `--impl 2`):

**Fixes RTL (right-to-left) issues in Hebrew subtitles:**
- 📍 **Correct punctuation positioning** (.!?,:;)
- 🔤 **Proper English word display** in Hebrew text
- 🔢 **Number direction fixes**
- 🎯 **Unicode RTL markers** for forced RTL mode

### Implementation Differences:

| Feature | impl=1 (Standard) | impl=2 (Advanced) | impl=2 + --fix-rtl |
|---------|------------------|------------------|-------------------|
| **Splitting** | By segment duration | By pauses + punctuation | By pauses + punctuation |
| **Hebrew** | Basic support | Special dash handling | + RTL fixes |
| **RTL Punctuation** | ❌ | ❌ | ✅ |
| **English words in Hebrew** | ❌ | ❌ | ✅ |
| **Line length** | Fixed splitting | Intelligent splitting | Intelligent splitting |
| **Quality** | Good | Excellent for Hebrew | Perfect for Hebrew |
| **Speed** | Faster | Slightly slower | Slightly slower |

### Parameters

- `--srt` - generate subtitles
- `--impl 1|2` - implementation choice (default 1)
- `--fix-rtl` - fix RTL issues (only for --impl 2 and Hebrew)
- `--lang he` - especially effective for impl=2 with Hebrew
- `--profile stable` - recommended for better quality

### Usage Examples

```bash
# Standard subtitles (fast)
python whisper_fullfile_cli.py --input lecture.mp3 --srt --impl 1 --lang he

# Advanced subtitles for Hebrew (high quality)
python whisper_fullfile_cli.py --input lecture.mp3 --srt --impl 2 --lang he --profile stable

# 🔥 MAXIMUM QUALITY for Hebrew with RTL fixes
python whisper_fullfile_cli.py --input lecture.mp3 --srt --impl 2 --lang he --fix-rtl --profile stable

# Default uses impl=1
python whisper_fullfile_cli.py --input audio.mp3 --srt --lang ru

# Result always: output.srt
```

### Technical Details

**Implementation 2** uses:
- **word_timestamps=True** to get per-word timestamps
- **ToSubs class** from mdb-ai project
- **pysrt library** for subtitle handling
- **Multi-level analysis**: pauses → punctuation → length

## Quality Profiles

- `--profile stable` (default) - best quality
- `--profile fast` - quick processing

## Supported Languages

- `he` - Hebrew (recommended for --impl 2)
- `ru` - Russian  
- `en` - English
- `he_old` - old Hebrew model 