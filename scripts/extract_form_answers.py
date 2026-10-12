#!/usr/bin/env python3
"""Mine AskUserQuestion decisions out of Claude Code transcripts.

Every Claude Code session lives as JSONL under ~/.claude/projects/<project>/.
An AskUserQuestion form shows up twice per call:

  * an assistant `tool_use` block carrying the questions and offered options
  * a user `tool_result` block carrying the rendered answer text

This joins them by tool_use_id and emits one record per form.

Synthetic (hook-satisfied) forms are detected, not guessed at. Claude Code
records a `hook_success` attachment on its own line, just before the result,
carrying `hookName: "PreToolUse:AskUserQuestion"` and the hook's stdout; when
that stdout rewrites the input with `updatedInput.answers`, no human was behind
the form. Those rows are flagged `synthetic: true` and must be excluded from
any dataset built to model the user's real choices.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_ROOT = Path(os.path.expanduser("~/.claude/projects"))

_ANSWERED = re.compile(r"answered:\s*(?P<body>.*?)(?:\.\s*Read the answers|\.?\s*$)", re.S)
_PAIR = re.compile(
    r'"(?P<key>(?:[^"\\]|\\.)*)"\s*=\s*(?P<value>"(?:[^"\\]|\\.)*"|\[[^\]]*\]|[^\s,][^,\n]*)'
)
_UNESCAPE = re.compile(r"\\(.)")


def _unescape(text: str) -> str:
    return _UNESCAPE.sub(r"\1", text)


def parse_answer_text(text: str) -> dict:
    """Pull {"question": answer} out of the rendered tool_result text."""
    if not isinstance(text, str) or "answered:" not in text:
        return {}
    match = _ANSWERED.search(text)
    if not match:
        return {}
    answers = {}
    for pair in _PAIR.finditer(match.group("body")):
        key = _unescape(pair.group("key"))
        raw = pair.group("value").strip()
        if raw.startswith("["):
            try:
                answers[key] = json.loads(raw)
                continue
            except ValueError:
                answers[key] = [p.strip().strip('"') for p in raw.strip("[]").split(",") if p.strip()]
                continue
        if raw.startswith('"'):
            answers[key] = _unescape(raw[1:-1])
        else:
            answers[key] = raw.rstrip(".")
    return answers


def _content_blocks(message) -> list:
    content = (message or {}).get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def iter_records(root: Path, project_filter: str | None = None):
    """Yield one dict per AskUserQuestion form found under root."""
    for path in sorted(root.rglob("*.jsonl")):
        project = path.parent.name
        if project_filter and project_filter.lower() not in project.lower():
            continue
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue

        with handle:
            forms: dict[str, dict] = {}
            synthetic_ids: set[str] = set()
            answered: list[dict] = []

            for line in handle:
                if "AskUserQuestion" not in line and "answered:" not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue

                attachment = row.get("attachment")
                if isinstance(attachment, dict):
                    hook_name = attachment.get("hookName") or ""
                    tool_id = attachment.get("toolUseID")
                    if hook_name.startswith("PreToolUse:AskUserQuestion") and tool_id:
                        stdout = attachment.get("stdout") or ""
                        if "updatedInput" in stdout and "answers" in stdout:
                            synthetic_ids.add(tool_id)

                for block in _content_blocks(row.get("message")):
                    kind = block.get("type")

                    if kind == "tool_use" and block.get("name") == "AskUserQuestion":
                        forms[block.get("id")] = {
                            "ts": row.get("timestamp"),
                            "session": path.stem,
                            "project": project,
                            "sidechain": bool(row.get("isSidechain")),
                            "questions": (block.get("input") or {}).get("questions") or [],
                        }

                    elif kind == "tool_result":
                        tool_use_id = block.get("tool_use_id")
                        form = forms.get(tool_use_id)
                        if form is None:
                            continue
                        text = _result_text(block)
                        raw_result = row.get("toolUseResult")
                        if isinstance(raw_result, str):
                            try:
                                raw_result = json.loads(raw_result)
                            except ValueError:
                                raw_result = None
                        structured = (
                            raw_result.get("answers") if isinstance(raw_result, dict) else None
                        )
                        if isinstance(structured, dict) and structured:
                            answers, source = structured, "structured"
                        else:
                            answers, source = parse_answer_text(text), "text"
                        answered.append(
                            {
                                **form,
                                "tool_use_id": tool_use_id,
                                "answers": answers,
                                "answer_source": source,
                                "options": [
                                    {
                                        "question": q.get("question"),
                                        "header": q.get("header"),
                                        "multiSelect": bool(q.get("multiSelect")),
                                        "labels": [
                                            o.get("label") for o in q.get("options") or []
                                        ],
                                    }
                                    for q in form["questions"]
                                ],
                            }
                        )

            for record in answered:
                record["synthetic"] = record["tool_use_id"] in synthetic_ids
                yield record


def main() -> int:
    # Transcripts are UTF-8; the Windows console codepage is not.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--project", default=None, help="substring filter on project dir name")
    parser.add_argument("--out", default=None, help="write records as JSONL")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    counts = {
        "total": 0,
        "synthetic": 0,
        "real": 0,
        "structured": 0,
        "text": 0,
        "empty": 0,
        "no_option_match": 0,
    }
    projects: dict[str, int] = {}
    answers_seen: dict[str, int] = {}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for record in iter_records(Path(args.root), args.project):
            counts["total"] += 1
            if record.get("synthetic"):
                counts["synthetic"] += 1
            else:
                counts["real"] += 1
            counts[record["answer_source"] if record["answers"] else "empty"] += 1
            projects[record["project"]] = projects.get(record["project"], 0) + 1

            labels = {l for opt in record["options"] for l in opt["labels"] if l}
            for question, answer in record["answers"].items():
                values = answer if isinstance(answer, list) else [answer]
                for value in values:
                    answers_seen[str(value)] = answers_seen.get(str(value), 0) + 1
                    if labels and str(value) not in labels:
                        counts["no_option_match"] += 1
            if out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if out:
            out.close()

    print(json.dumps(counts, indent=2))
    print("\ntop answers:")
    for value, n in sorted(answers_seen.items(), key=lambda kv: -kv[1])[:40]:
        print(f"  {n:5d}  {value[:100]}")
    print("\nprojects:")
    for name, n in sorted(projects.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {n:5d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
