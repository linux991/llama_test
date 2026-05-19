#!/usr/bin/env python3
"""Prepare CPU-only artifacts for the horror detection Colab experiment.

Inputs are cleaned one-story-per-file .txt files. Outputs are reports, a
story-level train/eval split, chunk CSVs with rule-based topic descriptions,
and a compact zip payload that can be uploaded to Google Colab.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import statistics
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EN_STOPWORDS = {
    "about", "after", "again", "against", "all", "also", "and", "any", "are",
    "around", "because", "been", "before", "being", "between", "both", "but",
    "came", "can", "could", "did", "didn", "does", "don", "down", "each",
    "even", "ever", "every", "for", "from", "get", "got", "had", "has",
    "have", "her", "here", "hers", "him", "his", "how", "into", "its",
    "just", "like", "little", "look", "made", "make", "many", "more",
    "most", "much", "must", "never", "not", "now", "off", "one", "only",
    "our", "out", "over", "said", "saw", "see", "she", "should", "some",
    "still", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "time", "too", "under",
    "until", "very", "was", "way", "were", "what", "when", "where",
    "which", "while", "who", "will", "with", "would", "you", "your",
}


@dataclass(frozen=True)
class Story:
    story_id: str
    title: str
    path: Path
    text: str
    n_chars: int
    n_words: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local CPU-only artifacts for the horror Colab experiment.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/creepypasta_stories_txt"),
        help="Directory with one cleaned .txt file per story.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for human-readable reports.",
    )
    parser.add_argument(
        "--payload-dir",
        type=Path,
        default=Path("data/colab_payload"),
        help="Directory for files that will be zipped for Colab.",
    )
    parser.add_argument(
        "--payload-zip",
        type=Path,
        default=Path("data/horror_experiment_payload.zip"),
        help="Output zip path for Colab upload.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=700)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument(
        "--max-train-chunks",
        type=int,
        default=2000,
        help="Cap train chunks included in the Colab payload. Use 0 for all.",
    )
    parser.add_argument(
        "--max-eval-chunks",
        type=int,
        default=500,
        help="Cap eval chunks included in the Colab payload. Use 0 for all.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z']*", text))


def title_from_filename(path: Path) -> str:
    stem = re.sub(r"^\d+_", "", path.stem)
    return stem.replace("_", " ").strip()


def load_stories(input_dir: Path) -> list[Story]:
    files = sorted(input_dir.glob("*.txt"))
    stories = []
    for index, path in enumerate(files, start=1):
        text = read_text(path)
        if not text:
            continue
        stories.append(
            Story(
                story_id=f"s{index:04d}",
                title=title_from_filename(path),
                path=path,
                text=text,
                n_chars=len(text),
                n_words=word_count(text),
                sha256=sha256_text(text),
            )
        )
    return stories


def split_stories(
    stories: list[Story],
    train_ratio: float,
    seed: int,
) -> tuple[list[Story], list[Story]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    shuffled = stories[:]
    random.Random(seed).shuffle(shuffled)
    split_at = int(len(shuffled) * train_ratio)
    return shuffled[:split_at], shuffled[split_at:]


def split_into_chunks(text: str, min_chars: int, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if len(current) >= min_chars:
                chunks.append(current)
            current = ""
            for start in range(0, len(sentence), max_chars):
                part = sentence[start : start + max_chars].strip()
                if len(part) >= min_chars:
                    chunks.append(part)
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if len(current) >= min_chars:
                chunks.append(current)
            current = sentence

    if len(current) >= min_chars:
        chunks.append(current)

    return chunks


def keywords(text: str, top_k: int = 8) -> list[str]:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z']*", text)
        if len(word) > 3 and word.lower() not in EN_STOPWORDS
    ]
    return [word for word, _count in Counter(words).most_common(top_k)]


def make_description(text: str, title: str) -> str:
    motifs = ", ".join(keywords(text)) or "darkness, fear, silence"
    clean_title = title.strip() or "untitled horror story"
    return (
        "Write a short literary horror fragment in English. "
        "Keep the prose natural and unsettling, with rising tension and no explanation. "
        f"Story context/title: {clean_title}. "
        f"Motifs or anchor words: {motifs}."
    )


def build_chunks(stories: list[Story], split: str, min_chars: int, max_chars: int) -> list[dict[str, object]]:
    rows = []
    for story in stories:
        chunks = split_into_chunks(story.text, min_chars, max_chars)
        for chunk_index, chunk in enumerate(chunks):
            rows.append(
                {
                    "split": split,
                    "story_id": story.story_id,
                    "source_file": story.path.name,
                    "title": story.title,
                    "chunk_id": chunk_index,
                    "text": chunk,
                    "description": make_description(chunk, story.title),
                    "n_chars": len(chunk),
                    "n_words": word_count(chunk),
                    "text_sha256": sha256_text(chunk),
                }
            )
    return rows


def sample_rows(rows: list[dict[str, object]], limit: int, seed: int) -> list[dict[str, object]]:
    if limit <= 0 or len(rows) <= limit:
        return rows[:]
    sampled = rows[:]
    random.Random(seed).shuffle(sampled)
    return sorted(sampled[:limit], key=lambda row: (str(row["story_id"]), int(row["chunk_id"])))


def remove_overlapping_eval_chunks(
    train_chunks: list[dict[str, object]],
    eval_chunks: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    train_hashes = {str(row["text_sha256"]) for row in train_chunks}
    filtered = [row for row in eval_chunks if str(row["text_sha256"]) not in train_hashes]
    return filtered, len(eval_chunks) - len(filtered)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100) * len(ordered)) - 1))
    return ordered[index]


def summarize_lengths(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "p95": 0, "max": 0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def write_corpus_report(
    path: Path,
    stories: list[Story],
    train_stories: list[Story],
    eval_stories: list[Story],
    train_chunks_full: list[dict[str, object]],
    eval_chunks_full: list[dict[str, object]],
    train_chunks_payload: list[dict[str, object]],
    eval_chunks_payload: list[dict[str, object]],
    leakage: dict[str, int],
    removed_eval_overlap_chunks: int,
    payload_zip: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    story_chars = summarize_lengths([story.n_chars for story in stories])
    story_words = summarize_lengths([story.n_words for story in stories])
    chunk_chars = summarize_lengths(
        [int(row["n_chars"]) for row in train_chunks_full + eval_chunks_full]
    )
    try:
        payload_zip_display = payload_zip.relative_to(Path.cwd())
    except ValueError:
        payload_zip_display = payload_zip

    lines = [
        "# Horror Corpus Preparation Report",
        "",
        "## Corpus",
        "",
        f"- Stories: {len(stories)}",
        f"- Train stories: {len(train_stories)}",
        f"- Eval stories: {len(eval_stories)}",
        f"- Story chars: min={story_chars['min']}, median={story_chars['median']:.0f}, mean={story_chars['mean']:.0f}, p95={story_chars['p95']}, max={story_chars['max']}",
        f"- Story words: min={story_words['min']}, median={story_words['median']:.0f}, mean={story_words['mean']:.0f}, p95={story_words['p95']}, max={story_words['max']}",
        "",
        "## Chunks",
        "",
        f"- Full train chunks: {len(train_chunks_full)}",
        f"- Full eval chunks: {len(eval_chunks_full)}",
        f"- Removed eval chunks due to exact train overlap: {removed_eval_overlap_chunks}",
        f"- Payload train chunks: {len(train_chunks_payload)}",
        f"- Payload eval chunks: {len(eval_chunks_payload)}",
        f"- Chunk chars: min={chunk_chars['min']}, median={chunk_chars['median']:.0f}, mean={chunk_chars['mean']:.0f}, p95={chunk_chars['p95']}, max={chunk_chars['max']}",
        "",
        "## Leakage Checks",
        "",
        f"- Overlapping story hashes between train/eval: {leakage['story_hash_overlap']}",
        f"- Overlapping chunk hashes between train/eval: {leakage['chunk_hash_overlap']}",
        "",
        "## Colab Payload",
        "",
        f"- Zip: `{payload_zip_display}`",
        "",
        "Payload files:",
        "",
        "- `train_with_descriptions.csv`",
        "- `eval_with_descriptions.csv`",
        "- `split_metadata.csv`",
        "- `corpus_summary.csv`",
        "- `corpus_preparation_report.md`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(path: Path, stories: list[Story], train_chunks: list[dict[str, object]], eval_chunks: list[dict[str, object]]) -> None:
    rows = [
        {"metric": "stories", "value": len(stories)},
        {"metric": "train_chunks_full", "value": len(train_chunks)},
        {"metric": "eval_chunks_full", "value": len(eval_chunks)},
        {"metric": "story_chars_total", "value": sum(story.n_chars for story in stories)},
        {"metric": "story_words_total", "value": sum(story.n_words for story in stories)},
    ]
    write_csv(path, rows, ["metric", "value"])


def build_payload(payload_dir: Path, payload_zip: Path, files: list[Path]) -> None:
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_zip.parent.mkdir(parents=True, exist_ok=True)
    if payload_zip.exists():
        payload_zip.unlink()
    with zipfile.ZipFile(payload_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, arcname=file_path.name)


def copy_for_payload(source: Path, payload_dir: Path) -> Path:
    target = payload_dir / source.name
    target.write_bytes(source.read_bytes())
    return target


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    reports_dir = args.reports_dir.resolve()
    payload_dir = args.payload_dir.resolve()
    payload_zip = args.payload_zip.resolve()

    stories = load_stories(input_dir)
    if not stories:
        raise ValueError(f"No .txt stories found in {input_dir}")

    train_stories, eval_stories = split_stories(stories, args.train_ratio, args.seed)
    train_chunks_full = build_chunks(train_stories, "train", args.min_chars, args.max_chars)
    eval_chunks_full = build_chunks(eval_stories, "eval", args.min_chars, args.max_chars)
    eval_chunks_full, removed_eval_overlap_chunks = remove_overlapping_eval_chunks(
        train_chunks_full,
        eval_chunks_full,
    )
    train_chunks_payload = sample_rows(train_chunks_full, args.max_train_chunks, args.seed)
    eval_chunks_payload = sample_rows(eval_chunks_full, args.max_eval_chunks, args.seed)

    story_train_hashes = {story.sha256 for story in train_stories}
    story_eval_hashes = {story.sha256 for story in eval_stories}
    chunk_train_hashes = {str(row["text_sha256"]) for row in train_chunks_full}
    chunk_eval_hashes = {str(row["text_sha256"]) for row in eval_chunks_full}
    leakage = {
        "story_hash_overlap": len(story_train_hashes & story_eval_hashes),
        "chunk_hash_overlap": len(chunk_train_hashes & chunk_eval_hashes),
    }

    chunk_fields = [
        "split",
        "story_id",
        "source_file",
        "title",
        "chunk_id",
        "text",
        "description",
        "n_chars",
        "n_words",
        "text_sha256",
    ]
    split_rows = [
        {
            "split": "train",
            "story_id": story.story_id,
            "source_file": story.path.name,
            "title": story.title,
            "n_chars": story.n_chars,
            "n_words": story.n_words,
            "story_sha256": story.sha256,
        }
        for story in train_stories
    ] + [
        {
            "split": "eval",
            "story_id": story.story_id,
            "source_file": story.path.name,
            "title": story.title,
            "n_chars": story.n_chars,
            "n_words": story.n_words,
            "story_sha256": story.sha256,
        }
        for story in eval_stories
    ]

    split_metadata_path = processed_dir / "split_metadata.csv"
    train_full_path = processed_dir / "train_chunks_full.csv"
    eval_full_path = processed_dir / "eval_chunks_full.csv"
    train_payload_path = processed_dir / "train_with_descriptions.csv"
    eval_payload_path = processed_dir / "eval_with_descriptions.csv"
    summary_csv_path = processed_dir / "corpus_summary.csv"
    report_path = reports_dir / "corpus_preparation_report.md"

    write_csv(
        split_metadata_path,
        split_rows,
        ["split", "story_id", "source_file", "title", "n_chars", "n_words", "story_sha256"],
    )
    write_csv(train_full_path, train_chunks_full, chunk_fields)
    write_csv(eval_full_path, eval_chunks_full, chunk_fields)
    write_csv(train_payload_path, train_chunks_payload, chunk_fields)
    write_csv(eval_payload_path, eval_chunks_payload, chunk_fields)
    write_summary_csv(summary_csv_path, stories, train_chunks_full, eval_chunks_full)

    write_corpus_report(
        report_path,
        stories,
        train_stories,
        eval_stories,
        train_chunks_full,
        eval_chunks_full,
        train_chunks_payload,
        eval_chunks_payload,
        leakage,
        removed_eval_overlap_chunks,
        payload_zip,
    )

    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_files = [
        copy_for_payload(train_payload_path, payload_dir),
        copy_for_payload(eval_payload_path, payload_dir),
        copy_for_payload(split_metadata_path, payload_dir),
        copy_for_payload(summary_csv_path, payload_dir),
        copy_for_payload(report_path, payload_dir),
    ]
    build_payload(payload_dir, payload_zip, payload_files)

    print(f"Stories: {len(stories)}")
    print(f"Train/eval stories: {len(train_stories)}/{len(eval_stories)}")
    print(f"Full train/eval chunks: {len(train_chunks_full)}/{len(eval_chunks_full)}")
    print(f"Removed eval chunks due to exact train overlap: {removed_eval_overlap_chunks}")
    print(f"Payload train/eval chunks: {len(train_chunks_payload)}/{len(eval_chunks_payload)}")
    print(f"Story hash overlap: {leakage['story_hash_overlap']}")
    print(f"Chunk hash overlap: {leakage['chunk_hash_overlap']}")
    print(f"Report: {report_path}")
    print(f"Payload zip: {payload_zip}")


if __name__ == "__main__":
    main()
