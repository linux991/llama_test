#!/usr/bin/env python3
"""Extract creepypasta stories from a Kaggle dataset dump into .txt files.

The script is intentionally conservative: it keeps one output file per source
story, filters empty/very short records, removes simple markup, deduplicates by
normalized text, and writes a metadata CSV for inspection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEXT_COLUMN_CANDIDATES = (
    "text",
    "story",
    "body",
    "content",
    "creepypasta",
    "post",
    "selftext",
    "description",
)

TITLE_COLUMN_CANDIDATES = (
    "title",
    "name",
    "story_name",
    "story_title",
    "post_title",
)


@dataclass
class Story:
    title: str
    text: str
    source_file: str
    source_row: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cleaned creepypasta stories from a Kaggle dataset into .txt files.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a Kaggle ZIP file, an extracted dataset directory, or a single CSV/JSON/TXT file.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/creepypasta_stories_txt"),
        type=Path,
        help="Directory where cleaned .txt stories will be written.",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        type=Path,
        help="Optional path for extraction metadata. Defaults to <output-dir>/metadata.csv.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="CSV/JSON column containing story text. If omitted, the script guesses.",
    )
    parser.add_argument(
        "--title-column",
        default=None,
        help="CSV/JSON column containing story titles. If omitted, the script guesses.",
    )
    parser.add_argument(
        "--min-chars",
        default=500,
        type=int,
        help="Drop stories shorter than this after cleaning.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing new files.",
    )
    return parser.parse_args()


def prepare_input(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="creepypasta_extract_")
        extract_dir = Path(tmp.name)
        with zipfile.ZipFile(input_path, "r") as archive:
            archive.extractall(extract_dir)
        return extract_dir, tmp

    return input_path, None


def discover_files(input_root: Path) -> list[Path]:
    supported = {".csv", ".json", ".jsonl", ".ndjson", ".txt", ".xlsx"}
    if input_root.is_file():
        return [input_root] if input_root.suffix.lower() in supported else []

    files = [
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    ]
    return sorted(files)


def read_text_safely(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def clean_text(raw_text: object) -> str:
    text = "" if raw_text is None else str(raw_text)
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def pick_column(
    fieldnames: Iterable[str],
    preferred: str | None,
    candidates: tuple[str, ...],
) -> str | None:
    fields = list(fieldnames)
    if preferred:
        if preferred not in fields:
            raise ValueError(f"Requested column '{preferred}' not found. Available: {fields}")
        return preferred

    lowered = {field.lower().strip(): field for field in fields}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    return None


def choose_longest_text_column(rows: list[dict[str, object]]) -> str | None:
    if not rows:
        return None

    scores: dict[str, int] = {}
    for key in rows[0].keys():
        values = [clean_text(row.get(key, "")) for row in rows[:100]]
        scores[key] = max((len(value) for value in values), default=0)

    best_key, best_score = max(scores.items(), key=lambda item: item[1])
    return best_key if best_score > 0 else None


def iter_csv_stories(
    path: Path,
    text_column: str | None,
    title_column: str | None,
) -> Iterable[Story]:
    text = read_text_safely(path)
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = list(reader)
    if not rows:
        return

    selected_text_column = pick_column(rows[0].keys(), text_column, TEXT_COLUMN_CANDIDATES)
    if selected_text_column is None:
        selected_text_column = choose_longest_text_column(rows)
    if selected_text_column is None:
        return

    selected_title_column = pick_column(rows[0].keys(), title_column, TITLE_COLUMN_CANDIDATES)

    for row_number, row in enumerate(rows, start=2):
        title = str(row.get(selected_title_column, "")).strip() if selected_title_column else ""
        yield Story(
            title=title,
            text=clean_text(row.get(selected_text_column, "")),
            source_file=str(path),
            source_row=row_number,
        )


def iter_json_records(data: object) -> Iterable[dict[str, object]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield data


def iter_json_stories(
    path: Path,
    text_column: str | None,
    title_column: str | None,
) -> Iterable[Story]:
    raw = read_text_safely(path)
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        records = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    else:
        records = list(iter_json_records(json.loads(raw)))

    if not records:
        return

    selected_text_column = pick_column(records[0].keys(), text_column, TEXT_COLUMN_CANDIDATES)
    if selected_text_column is None:
        selected_text_column = choose_longest_text_column(records)
    if selected_text_column is None:
        return

    selected_title_column = pick_column(records[0].keys(), title_column, TITLE_COLUMN_CANDIDATES)

    for index, row in enumerate(records, start=1):
        title = str(row.get(selected_title_column, "")).strip() if selected_title_column else ""
        yield Story(
            title=title,
            text=clean_text(row.get(selected_text_column, "")),
            source_file=str(path),
            source_row=index,
        )


def iter_txt_story(path: Path) -> Iterable[Story]:
    yield Story(
        title=path.stem,
        text=clean_text(read_text_safely(path)),
        source_file=str(path),
        source_row=None,
    )


def read_xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for item in root.findall("x:si", namespace):
        parts = [node.text or "" for node in item.findall(".//x:t", namespace)]
        strings.append("".join(parts))
    return strings


def read_xlsx_first_sheet_rows(path: Path) -> list[list[object]]:
    with zipfile.ZipFile(path, "r") as archive:
        shared_strings = read_xlsx_shared_strings(archive)
        sheet_names = [
            name
            for name in archive.namelist()
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        ]
        if not sheet_names:
            return []

        root = ET.fromstring(archive.read(sorted(sheet_names)[0]))
        namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows: list[list[object]] = []

        for row in root.findall(".//x:sheetData/x:row", namespace):
            values_by_col: dict[int, object] = {}
            for cell in row.findall("x:c", namespace):
                cell_ref = cell.attrib.get("r", "")
                match = re.match(r"([A-Z]+)", cell_ref)
                if not match:
                    continue

                col_index = column_letters_to_index(match.group(1))
                cell_type = cell.attrib.get("t")
                value_node = cell.find("x:v", namespace)
                inline_node = cell.find("x:is/x:t", namespace)

                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text or 0)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""

                values_by_col[col_index] = value

            if values_by_col:
                max_col = max(values_by_col)
                rows.append([values_by_col.get(i, "") for i in range(max_col + 1)])

        return rows


def column_letters_to_index(letters: str) -> int:
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def iter_xlsx_stories(
    path: Path,
    text_column: str | None,
    title_column: str | None,
) -> Iterable[Story]:
    rows = read_xlsx_first_sheet_rows(path)
    if len(rows) < 2:
        return

    headers = [str(value).strip() for value in rows[0]]
    records = []
    for row in rows[1:]:
        record = {headers[index]: row[index] if index < len(row) else "" for index in range(len(headers))}
        records.append(record)

    if not records:
        return

    selected_text_column = pick_column(headers, text_column, TEXT_COLUMN_CANDIDATES)
    if selected_text_column is None:
        selected_text_column = choose_longest_text_column(records)
    if selected_text_column is None:
        return

    selected_title_column = pick_column(headers, title_column, TITLE_COLUMN_CANDIDATES)

    for row_number, row in enumerate(records, start=2):
        title = str(row.get(selected_title_column, "")).strip() if selected_title_column else ""
        yield Story(
            title=title,
            text=clean_text(row.get(selected_text_column, "")),
            source_file=str(path),
            source_row=row_number,
        )


def iter_stories(
    files: Iterable[Path],
    text_column: str | None,
    title_column: str | None,
) -> Iterable[Story]:
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            yield from iter_csv_stories(path, text_column, title_column)
        elif suffix in {".json", ".jsonl", ".ndjson"}:
            yield from iter_json_stories(path, text_column, title_column)
        elif suffix == ".txt":
            yield from iter_txt_story(path)
        elif suffix == ".xlsx":
            yield from iter_xlsx_stories(path, text_column, title_column)


def slugify(value: str, fallback: str) -> str:
    value = value.strip() or fallback
    value = re.sub(r"[^\w\s.-]+", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._-")
    return value[:80] or fallback


def write_outputs(
    stories: Iterable[Story],
    output_dir: Path,
    metadata_csv: Path,
    min_chars: int,
) -> tuple[int, int, int]:
    seen_hashes: set[str] = set()
    written = 0
    skipped_short = 0
    skipped_duplicate = 0
    metadata_rows = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for story in stories:
        if len(story.text) < min_chars:
            skipped_short += 1
            continue

        digest = normalized_hash(story.text)
        if digest in seen_hashes:
            skipped_duplicate += 1
            continue
        seen_hashes.add(digest)

        written += 1
        title = story.title or f"story_{written:04d}"
        filename = f"{written:04d}_{slugify(title, f'story_{written:04d}')}.txt"
        output_path = output_dir / filename
        output_path.write_text(story.text + "\n", encoding="utf-8")

        metadata_rows.append(
            {
                "story_id": written,
                "filename": filename,
                "title": story.title,
                "n_chars": len(story.text),
                "sha256": digest,
                "source_file": story.source_file,
                "source_row": story.source_row,
            }
        )

    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "story_id",
                "filename",
                "title",
                "n_chars",
                "sha256",
                "source_file",
                "source_row",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    return written, skipped_short, skipped_duplicate


def main() -> None:
    args = parse_args()
    input_root, tmp = prepare_input(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    metadata_csv = (
        args.metadata_csv.expanduser().resolve()
        if args.metadata_csv
        else output_dir / "metadata.csv"
    )

    try:
        if args.overwrite and output_dir.exists():
            shutil.rmtree(output_dir)

        files = discover_files(input_root)
        if not files:
            raise ValueError(f"No supported dataset files found under: {input_root}")

        stories = iter_stories(files, args.text_column, args.title_column)
        written, skipped_short, skipped_duplicate = write_outputs(
            stories=stories,
            output_dir=output_dir,
            metadata_csv=metadata_csv,
            min_chars=args.min_chars,
        )

        print(f"Supported source files found: {len(files)}")
        print(f"Stories written: {written}")
        print(f"Skipped as too short: {skipped_short}")
        print(f"Skipped as duplicates: {skipped_duplicate}")
        print(f"Output directory: {output_dir}")
        print(f"Metadata CSV: {metadata_csv}")
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
