#!/usr/bin/env python3
"""
Ops Meeting AI Agent (Malayalam → English)
==========================================

Inspired by the simple agent pattern (Task/Tool/Agent) in TechWithTim's
"PythonAIAgentFromScratch" project, this single-file implementation turns a
Microsoft Teams recording of a Malayalam operations meeting into a structured,
actionable English summary.

Key features
------------
1) Audio extraction from video (via ffmpeg)
2) Speech-to-text with Malayalam→English translation using any of:
   - OpenAI Whisper API (cloud)
   - Faster-Whisper (local)
   - HF Whisper-medium Malayalam fine-tune (local) → translate via LLM
   - HF Wav2Vec2 Malayalam (local CTC) → translate via LLM
3) Register transformation: polished business English (LLM)
4) Perspective-wise summarization by issue: highlights, decisions, action items
5) Clean, structured Markdown output that you can copy into docs/emails

Usage
-----
# 1) Install dependencies
#    pip install -r requirements.txt
#    (Also install system ffmpeg: https://ffmpeg.org/ and ensure it is on PATH)
#
# 2) Environment
#    OPENAI_API_KEY=sk-...            (required for LLM and for Whisper API mode)
#    OPENAI_BASE_URL=https://api.openai.com/v1   (optional; set if using a proxy)
#    LLM_MODEL=gpt-4o-mini            (or gpt-4o, etc.)
#    STT_BACKEND=openai|faster_whisper|hf_whisper_ml|hf_wav2vec2_ml
#    FW_MODEL=large-v3                (for faster-whisper; or medium/small)
#    HF_WHISPER_ML_MODEL=vrclc/Whisper-medium-Malayalam
#    HF_W2V2_ML_MODEL=gvs/wav2vec2-large-xlsr-malayalam
#    HF_MLXLMR_MODEL=bytesizedllm/MalayalamXLM_Roberta
#
# 3) Run (new dual-output):
#   python main.py \
#       --input "TeamsRecording.mp4" \
#       --date "2025-08-10" \
#       --attendees "Rahul (Installation); Meera (Production); Akash (Design)" \
#       --out-summary "meeting_summary.md" \
#       --out-transcript "meeting_transcript_en.txt" \
#       --stt openai
#
# 4) Backward-compatible (summary only):
#   python main.py --input "file.mp4" --date "2025-08-10" --attendees "Rahul; Meera" --out "summary.md"
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# -------------------------
# Domain data structures
# -------------------------
DEPARTMENTS = [
    "Installation", "Production", "Design", "Estimation",
    "Quality", "Logistics", "Stores"
]


class ActionItem(BaseModel):
    description: str
    owner: str = Field(description="Person or department responsible")
    deadline: Optional[str] = Field(default=None, description="Date or milestone")


class IssueSummary(BaseModel):
    issue_name: str
    highlights: List[str]
    decisions: List[str]
    action_items: List[ActionItem]


class MeetingSummary(BaseModel):
    meeting_date: str
    attendees: List[str]
    purpose: str
    issues: List[IssueSummary]


# -------------------------
# Utilities
# -------------------------
def sh(cmd: List[str]) -> None:
    """Run a shell command and raise if it fails."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )


def ensure_ffmpeg() -> None:
    try:
        proc = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError
    except Exception:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and make sure it is on PATH.")


# -------------------------
# Agent pattern primitives (orchestrator)
# -------------------------
@dataclass
class Task:
    name: str
    description: str


class Tool:
    name: str = "tool"
    description: str = ""

    def run(self, **kwargs) -> Any:
        raise NotImplementedError


class Agent:
    """Simple orchestrator that chains tools with a context dict."""
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools

    def run(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        for step in plan:
            tool_name = step["tool"]
            inputs = step.get("inputs", {})
            # Resolve $ctx placeholders
            for k, v in list(inputs.items()):
                if isinstance(v, str) and v.startswith("$ctx."):
                    ctx_key = v.split(".", 1)[1]
                    inputs[k] = context.get(ctx_key)
            out_key = step.get("save_as")
            result = self.tools[tool_name].run(**inputs)
            if out_key:
                context[out_key] = result
        return context


# -------------------------
# Tools
# -------------------------
class AudioExtractor(Tool):
    name = "audio_extractor"
    description = "Extracts WAV audio from a video file using ffmpeg."

    def run(self, input_video: str, out_wav: Optional[str] = None, sr: int = 16000) -> str:
        ensure_ffmpeg()
        if out_wav is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            out_wav = tmp.name
            tmp.close()
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-ac", "1",        # mono
            "-ar", str(sr),    # sample rate
            out_wav
        ]
        sh(cmd)
        return out_wav


class Transcriber(Tool):
    name = "transcriber"
    description = "Transcribes Malayalam speech to English using OpenAI/Faster-Whisper/HF models."

    def __init__(
        self,
        backend: Literal["openai", "faster_whisper", "hf_whisper_ml", "hf_wav2vec2_ml"] = "openai",
        fw_model: str = "large-v3",
    ):
        self.backend = backend
        self.fw_model = fw_model

    def run(self, wav_path: str) -> str:
        if self.backend == "openai":
            return self._run_openai_whisper(wav_path)
        elif self.backend == "faster_whisper":
            return self._run_faster_whisper(wav_path)
        elif self.backend == "hf_whisper_ml":
            return self._run_hf_whisper_malayalam(wav_path)
        elif self.backend == "hf_wav2vec2_ml":
            return self._run_hf_wav2vec2_malayalam(wav_path)
        else:
            raise ValueError(f"Unknown STT backend: {self.backend}")

    def _run_openai_whisper(self, wav_path: str) -> str:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI Whisper mode.")
        base_url = os.getenv("OPENAI_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        model = os.getenv("WHISPER_MODEL", "gpt-4o-transcribe")

        with open(wav_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=f,
                translate=True,
                response_format="text",
            )
        return transcript

    def _run_faster_whisper(self, wav_path: str) -> str:
        from faster_whisper import WhisperModel

        device = "auto"
        compute_type = os.getenv("FW_COMPUTE_TYPE", "int8_float16")
        model_name = os.getenv("FW_MODEL", self.fw_model)
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        segments, _ = model.transcribe(
            wav_path,
            language="ml",
            task="translate",
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        return " ".join([seg.text.strip() for seg in segments if seg.text])

    def _run_hf_whisper_malayalam(self, wav_path: str) -> str:
        import os
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import soundfile as sf
        import librosa
        import numpy as np
        import gc

        model_id = os.getenv("HF_WHISPER_ML_MODEL", "vrclc/Whisper-medium-Malayalam")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        model.eval()
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ml", task="transcribe")

        target_sr = 16000
        chunk_seconds = int(os.getenv("WHISPER_CHUNK_SECONDS", "30"))  # lower if OOM persists
        chunk_size = chunk_seconds * target_sr

        out_en = []

        with sf.SoundFile(wav_path, 'r') as f:
            src_sr = f.samplerate
            total_frames = len(f)
            read_pos = 0

            while read_pos < total_frames:
                frames_to_read = min(chunk_size, total_frames - read_pos)
                f.seek(read_pos)
                block = f.read(frames_to_read, dtype='float32')
                read_pos += frames_to_read

                if block.ndim > 1:
                    block = block.mean(axis=1)

                if src_sr != target_sr:
                    block = librosa.resample(block, orig_sr=src_sr, target_sr=target_sr)

                if block.size == 0:
                    continue

                # Generate Malayalam text for this chunk
                with torch.no_grad():
                    inputs = processor(block, sampling_rate=target_sr, return_tensors="pt").to(device)
                    generated_ids = model.generate(**inputs)
                    ml_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                if ml_text:
                    # translate chunk
                    translation_prompt = (
                        "Translate the following Malayalam meeting transcript chunk into clear, idiomatic English:\n\n"
                        + ml_text
                    )
                    en_chunk = llm_complete(translation_prompt, model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
                    out_en.append(en_chunk)

                # free memory
                del inputs, generated_ids
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        return " ".join(out_en).strip()

    def _run_hf_wav2vec2_malayalam(self, wav_path: str) -> str:
        import os
        import math
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import soundfile as sf
        import numpy as np
        import librosa
        import gc

        model_id = os.getenv("HF_W2V2_ML_MODEL", "gvs/wav2vec2-large-xlsr-malayalam")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
        model.eval()

        # --- chunking params ---
        target_sr = 16000
        chunk_seconds = int(os.getenv("W2V2_CHUNK_SECONDS", "30"))  # reduce to 10-20 if still OOM
        hop_seconds = chunk_seconds  # non-overlapping; you can overlap slightly if you like
        chunk_size = chunk_seconds * target_sr
        hop_size = hop_seconds * target_sr

        # iterate using a streaming reader to avoid loading entire file
        with sf.SoundFile(wav_path, 'r') as f:
            src_sr = f.samplerate
            total_frames = len(f)
            out_text = []
            read_pos = 0

            while read_pos < total_frames:
                # read a block of frames
                frames_to_read = min(hop_size, total_frames - read_pos)
                f.seek(read_pos)
                block = f.read(frames_to_read, dtype='float32')
                read_pos += frames_to_read

                # ensure mono
                if block.ndim > 1:
                    block = block.mean(axis=1)

                # resample to 16k if necessary
                if src_sr != target_sr:
                    block = librosa.resample(block, orig_sr=src_sr, target_sr=target_sr)

                if block.size == 0:
                    continue

                # run model on this chunk
                with torch.no_grad():
                    inputs = processor(block, sampling_rate=target_sr, return_tensors="pt", padding="longest")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    logits = model(**inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    ml_text = processor.batch_decode(predicted_ids)[0].strip()

                # translate Malayalam -> English via LLM
                if ml_text:
                    translation_prompt = (
                        "Translate the following Malayalam meeting transcript chunk into clear, idiomatic English:\n\n"
                        + ml_text
                    )
                    en_chunk = llm_complete(translation_prompt, model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
                    out_text.append(en_chunk)

                # free memory incrementally
                del inputs, logits, predicted_ids
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        return " ".join(out_text).strip()

class MalayalamIssueExtractor(Tool):
    name = "ml_issue_extractor"
    description = "(Optional) Uses a local Malayalam LLaMA to chunk transcript into issues. Placeholder."

    def run(self, malayalam_transcript: str) -> str:
        # Placeholder for future local Malayalam LLM segmentation flow.
        return malayalam_transcript


class RegisterTransformer(Tool):
    name = "register_transformer"
    description = "Cleans disfluencies and rewrites transcript into polished business English."

    def __init__(self, model_env_var: str = "LLM_MODEL"):
        self.model_name = os.getenv(model_env_var, "gpt-4o-mini")

    def run(self, raw_english_transcript: str) -> str:
        prompt = f"""
You are a corporate communications editor. Rewrite the following meeting transcript
into concise, polished business English suitable for an executive audience. Follow
these rules:
- Remove filler words and false starts.
- Standardize corporate terminology (e.g., "deployment" instead of "setting up").
- Fix grammar and punctuation.
- Preserve meaning; do not omit key details.
- Keep speaker labels out; we only need content.

Transcript:
\"\"\"
{raw_english_transcript}
\"\"\""""
        return llm_complete(prompt, model=self.model_name)


class DeptCueTagger(Tool):
    name = "dept_cue_tagger"
    description = "Scores sentences against Ops departments using Malayalam-XLM-R embeddings."

    def __init__(self):
        self.model_id = os.getenv("HF_MLXLMR_MODEL", "bytesizedllm/MalayalamXLM_Roberta")

    def run(self, transcript_english: str) -> Dict[str, Any]:
        # Simple cosine-similarity scoring on CLS embeddings to hint owners
        from transformers import AutoTokenizer, AutoModel
        import torch
        import torch.nn.functional as F

        tok = AutoTokenizer.from_pretrained(self.model_id)
        mdl = AutoModel.from_pretrained(self.model_id)
        mdl.eval()

        def embed(texts: List[str]):
            with torch.no_grad():
                out = mdl(**tok(texts, padding=True, truncation=True, return_tensors="pt"))
                return out.last_hidden_state[:, 0]  # CLS

        # crude sentence split
        sents = [s.strip() for s in transcript_english.replace("\n", " ").split(".") if s.strip()]
        if not sents:
            return {"annotations": []}
        sent_emb = embed(sents)
        dept_emb = embed(DEPARTMENTS)

        sims = torch.nn.functional.cosine_similarity(
            sent_emb.unsqueeze(1), dept_emb.unsqueeze(0), dim=-1
        )  # [S, D]
        best = sims.argmax(dim=1).tolist()
        scores = sims.max(dim=1).values.tolist()

        annotations = []
        for i, s in enumerate(sents):
            annotations.append({
                "sentence": s,
                "dept": DEPARTMENTS[best[i]],
                "score": float(scores[i]),
            })
        return {"annotations": annotations}


class IssueSummarizer(Tool):
    name = "issue_summarizer"
    description = "Extracts issues, highlights, decisions, and action items with owners and deadlines."

    def __init__(self, model_env_var: str = "LLM_MODEL"):
        self.model_name = os.getenv(model_env_var, "gpt-4o-mini")

    def run(
        self,
        polished_transcript: str,
        meeting_date: str,
        attendees: List[str],
        dept_cues: Optional[Dict[str, Any]] = None
    ) -> MeetingSummary:
        cue_hint = ""
        if dept_cues and dept_cues.get("annotations"):
            # Provide a compact hint to steer owner assignment
            top_examples = dept_cues["annotations"][:30]
            cue_pairs = [f"[{a['dept']}] {a['sentence']}" for a in top_examples]
            cue_hint = "\n\nOwner hints (dept cues):\n" + "\n".join(cue_pairs)

        sys_instructions = f"""
You are an operations analyst. From the transcript, identify distinct issues
raised and discussants’ points. For each issue, provide:
- issue_name: short, generalized title
- highlights: 3–8 bullets of core discussion points
- decisions: 1–5 concrete decisions
- action_items: list of action items with owner (person or department) and deadline

Guidelines:
- Owners: If a specific person is unclear, assign the most relevant department
  from this set: {', '.join(DEPARTMENTS)}.
- Deadlines: If none are stated, infer a reasonable milestone (e.g., "next sprint",
  "by EOW", "before site handover") and note it as an inferred deadline.
- Keep content concise and businesslike.
- Do NOT include speaker names.
- Output strictly valid JSON for the schema below.

JSON schema:
{{
  "meeting_date": "{meeting_date}",
  "attendees": {attendees},
  "purpose": "<one-sentence purpose>",
  "issues": [
    {{
      "issue_name": "<string>",
      "highlights": ["<string>", ...],
      "decisions": ["<string>", ...],
      "action_items": [
        {{"description": "<string>", "owner": "<string>", "deadline": "<string|null>"}},
        ...
      ]
    }}, ...
  ]
}}
"""
        user_prompt = f"TRANSCRIPT:\n\n{polished_transcript}{cue_hint}\n\nReturn ONLY the JSON object."
        raw_json = llm_complete(system=sys_instructions, user=user_prompt, model=self.model_name)
        data = json.loads(raw_json)
        return MeetingSummary(**data)


class MarkdownFormatter(Tool):
    name = "markdown_formatter"
    description = "Renders MeetingSummary (pydantic) into Markdown per the requested format."

    def run(self, summary: MeetingSummary) -> str:
        lines: List[str] = []
        lines.append("# Operations Meeting Summary\n")
        lines.append(f"**Date:** {summary.meeting_date}\n")
        lines.append(f"**Attendees:** {', '.join(summary.attendees)}\n")
        lines.append("\n")
        lines.append(f"**Purpose:** {summary.purpose}\n")
        lines.append("\n")
        lines.append("## Issues Raised\n")
        for issue in summary.issues:
            lines.append(f"### {issue.issue_name}\n")
            if issue.highlights:
                lines.append("**Discussion Highlights**")
                for h in issue.highlights:
                    lines.append(f"- {h}")
                lines.append("\n")
            if issue.decisions:
                lines.append("**Decisions**")
                for d in issue.decisions:
                    lines.append(f"- {d}")
                lines.append("\n")
            if issue.action_items:
                lines.append("**Action Items**")
                for ai in issue.action_items:
                    owner = ai.owner
                    deadline = f" (Deadline: {ai.deadline})" if ai.deadline else ""
                    lines.append(f"- **{ai.description}** — **Owner:** {owner}{deadline}")
                lines.append("\n")
        return "\n".join(lines).strip() + "\n"


# -------------------------
# LLM helper
# -------------------------
def llm_complete(
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    user: Optional[str] = None
) -> str:
    """OpenAI-compatible Chat Completions helper.
    - Set OPENAI_API_KEY, optionally OPENAI_BASE_URL
    - Provide either a single combined `prompt` (as user) or system+user messages.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if prompt and not user:
        messages.append({"role": "user", "content": prompt})
    else:
        if user:
            messages.append({"role": "user", "content": user})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# -------------------------
# Main CLI orchestration
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Ops Meeting AI Agent (Malayalam → English)")
    parser.add_argument("--input", required=True, help="Path to Teams video file (e.g., .mp4)")
    parser.add_argument("--date", required=True, help="Meeting date, e.g., 2025-08-10")
    parser.add_argument(
        "--attendees",
        required=True,
        help="Semicolon-separated names, e.g., 'Rahul; Meera; Akash' or 'Rahul (Installation); ...'",
    )
    # --- output args ---
    parser.add_argument("--out", default=None, help="(Legacy) Output Markdown summary path")  # >>> CHANGED (was default file)
    parser.add_argument("--out-summary", default=None, help="Output Markdown summary path")   # >>> NEW
    parser.add_argument("--out-transcript", default=None, help="Output FULL English transcript (TXT)")  # >>> NEW

    parser.add_argument(
        "--stt",
        default=os.getenv("STT_BACKEND", "openai"),
        choices=["openai", "faster_whisper", "hf_whisper_ml", "hf_wav2vec2_ml"],
        help="Speech-to-text backend",
    )
    parser.add_argument("--tmp_wav", default=None, help="Optional path for extracted WAV (debugging)")

    args = parser.parse_args()

    # Validate output intent  # >>> NEW
    dual_requested = bool(args.out_summary or args.out_transcript)
    legacy_requested = bool(args.out)

    if not dual_requested and not legacy_requested:
        raise SystemExit(
            "Please specify either --out (legacy summary only) "
            "or --out-summary/--out-transcript for dual-output."
        )

    attendees = [a.strip() for a in args.attendees.split(";") if a.strip()]

    tools: Dict[str, Tool] = {
        "audio_extractor": AudioExtractor(),
        "transcriber": Transcriber(backend=args.stt),
        "register_transformer": RegisterTransformer(),
        "dept_cue_tagger": DeptCueTagger(),
        "issue_summarizer": IssueSummarizer(),
        "markdown_formatter": MarkdownFormatter(),
    }

    agent = Agent(tools)

    plan = [
        {"tool": "audio_extractor", "inputs": {"input_video": args.input, "out_wav": args.tmp_wav}, "save_as": "wav"},
        {"tool": "transcriber", "inputs": {"wav_path": "$ctx.wav"}, "save_as": "raw_en"},
        {"tool": "register_transformer", "inputs": {"raw_english_transcript": "$ctx.raw_en"}, "save_as": "polished"},
        {"tool": "dept_cue_tagger", "inputs": {"transcript_english": "$ctx.polished"}, "save_as": "dept_cues"},
        {
            "tool": "issue_summarizer",
            "inputs": {
                "polished_transcript": "$ctx.polished",
                "meeting_date": args.date,
                "attendees": attendees,
                "dept_cues": "$ctx.dept_cues",
            },
            "save_as": "summary",
        },
        {"tool": "markdown_formatter", "inputs": {"summary": "$ctx.summary"}, "save_as": "markdown"},
    ]

    ctx = agent.run(plan)

    summary_md: str = ctx["markdown"]
    full_english_transcript: str = ctx["raw_en"]  # >>> CHANGED: raw (unedited) English transcript

    # --- Write outputs ---  # >>> NEW
    if dual_requested:
        if args.out_summary:
            with open(args.out_summary, "w", encoding="utf-8") as f:
                f.write(summary_md)
            print(f"Summary written to: {args.out_summary}")
        if args.out_transcript:
            with open(args.out_transcript, "w", encoding="utf-8") as f:
                f.write(full_english_transcript)
            print(f"Full English transcript written to: {args.out_transcript}")

    # Legacy path: write only summary to --out  # >>> CHANGED
    if legacy_requested:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(summary_md)
        print(f"(Legacy) Summary written to: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        sys.exit(1)
