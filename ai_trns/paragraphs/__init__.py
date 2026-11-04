"""
Paragraphs of a lesson: who speaks (the diarize document) + where the Rav's speech breaks into paragraphs.

    speaker turn  = a run of words of one speaker; a speaker change always starts a new paragraph
    Rav's turns   = split further by the paragraphing model (mdb-lora: Qwen2.5-3B + LoRA "paragraphing",
                    served on `stt` by service.py, called by the diarize worker through client.py)

The diarize worker stores the model's paragraph starts in its document as word indices:

    doc["paragraph_starts"] = [i, ...]   # words[i] starts a paragraph inside a Rav turn
    doc["stats"]["paragraphs_error"]     # set if the service could not be used (then turns only)

build.py turns that document into the "paragraphs" result format (CPU, stdlib only):

    {"paragraphs": [{"speaker", "label", "start", "end", "text", "words": [{"word", "start", "end"}]}],
     "stats": {"paragraphs", "turns", "model_breaks"}}
"""
