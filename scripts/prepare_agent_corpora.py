# scripts/prepare_agent_corpora.py

import csv
import json
import re
from pathlib import Path
from typing import Any


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_CSV = Path(
    "data/cleaned/ww1_exploration_english_candidates.csv"
)

OUTPUT_DIR = Path("data/processed/agent_corpora")

STORY_OUTPUT = OUTPUT_DIR / "story_agent_documents.jsonl"
MATERIALS_OUTPUT = OUTPUT_DIR / "materials_agent_documents.jsonl"
METADATA_OUTPUT = OUTPUT_DIR / "metadata_agent_documents.jsonl"
BASELINE_OUTPUT = OUTPUT_DIR / "single_agent_documents.jsonl"
MANIFEST_OUTPUT = OUTPUT_DIR / "corpus_manifest.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "corpus_summary.json"

# Keep records with a meaningful main narrative.
MIN_STORY_WORDS = 40

# Keep a materials document only when it contains more than a tiny label.
MIN_MATERIALS_WORDS = 5


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def clean_text(value: Any) -> str:
    """Convert a value into clean, single-spaced text."""
    if value is None:
        return ""

    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    """Count words in a text string."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def split_pipe_values(value: str) -> list[str]:
    """
    Split CSV fields such as:
    '1914 | 1918'
    into:
    ['1914', '1918']
    """
    cleaned = clean_text(value)

    if not cleaned:
        return []

    return [
        part.strip()
        for part in cleaned.split("|")
        if part.strip()
    ]


def make_document(
    *,
    document_id: str,
    record_id: str,
    agent: str,
    source_field: str,
    text: str,
    title: str,
    europeana_url: str,
) -> dict[str, str]:
    """Create one consistent JSONL document."""
    return {
        "document_id": document_id,
        "record_id": record_id,
        "agent": agent,
        "source_field": source_field,
        "title": title,
        "text": text,
        "europeana_url": europeana_url,
    }


def write_jsonl(path: Path, documents: list[dict[str, str]]) -> None:
    """Write a list of dictionaries as JSON Lines."""
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(
                json.dumps(document, ensure_ascii=False) + "\n"
            )


def write_manifest(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write a CSV recording what was included in each corpus."""
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# Main processing
# --------------------------------------------------

def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "Could not find the cleaned candidate CSV:\n"
            f"{INPUT_CSV.resolve()}\n\n"
            "Run normalise_europeana.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading:")
    print(INPUT_CSV.resolve())

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        records = list(reader)

    if not records:
        raise RuntimeError(
            "The candidate CSV exists, but it has no rows."
        )

    story_documents: list[dict[str, str]] = []
    materials_documents: list[dict[str, str]] = []
    metadata_documents: list[dict[str, str]] = []
    baseline_documents: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    excluded_short_story = 0

    for row in records:
        record_id = clean_text(row.get("record_id"))
        title = clean_text(row.get("title_variants"))
        europeana_url = clean_text(row.get("europeana_url"))

        story_text = clean_text(
            row.get("english_longest_description")
        )

        materials_text = clean_text(
            row.get("english_other_descriptions")
        )

        # The main story text is required for a record to enter
        # the final experimental corpus.
        if word_count(story_text) < MIN_STORY_WORDS:
            excluded_short_story += 1

            manifest_rows.append(
                {
                    "record_id": record_id,
                    "title": title,
                    "included_in_final_corpus": "False",
                    "reason": (
                        f"Main story has fewer than "
                        f"{MIN_STORY_WORDS} words"
                    ),
                    "story_words": str(word_count(story_text)),
                    "materials_words": str(word_count(materials_text)),
                }
            )

            continue

        # -----------------------------
        # 1. Story-agent corpus
        # -----------------------------
        story_document = make_document(
            document_id=f"{record_id}::story",
            record_id=record_id,
            agent="story",
            source_field="english_longest_description",
            text=story_text,
            title=title,
            europeana_url=europeana_url,
        )

        story_documents.append(story_document)

        # -----------------------------
        # 2. Materials-agent corpus
        # -----------------------------
        has_materials = (
            word_count(materials_text) >= MIN_MATERIALS_WORDS
        )

        if has_materials:
            materials_document = make_document(
                document_id=f"{record_id}::materials",
                record_id=record_id,
                agent="materials",
                source_field="english_other_descriptions",
                text=materials_text,
                title=title,
                europeana_url=europeana_url,
            )

            materials_documents.append(materials_document)

        # -----------------------------
        # 3. Metadata-agent corpus
        # -----------------------------
        years = ", ".join(
            split_pipe_values(row.get("year_values", ""))
        )

        places = ", ".join(
            split_pipe_values(row.get("place_values", ""))
        )

        subjects = ", ".join(
            split_pipe_values(row.get("english_subjects", ""))
        )

        original_languages = ", ".join(
            split_pipe_values(
                row.get("original_item_language", "")
            )
        )

        record_type = clean_text(row.get("record_type"))
        rights = clean_text(row.get("rights"))
        provider = clean_text(row.get("data_provider"))

        metadata_lines = [
            f"Title: {title}",
            f"Record type: {record_type}",
            f"Years: {years}",
            f"Places: {places}",
            f"Subjects: {subjects}",
            f"Original item language: {original_languages}",
            f"Provider: {provider}",
            f"Rights: {rights}",
        ]

        # Remove lines that end with a blank value.
        metadata_text = "\n".join(
            line
            for line in metadata_lines
            if not line.endswith(": ")
        )

        metadata_document = make_document(
            document_id=f"{record_id}::metadata",
            record_id=record_id,
            agent="metadata",
            source_field="structured_metadata",
            text=metadata_text,
            title=title,
            europeana_url=europeana_url,
        )

        metadata_documents.append(metadata_document)

        # -----------------------------
        # 4. Single-agent baseline corpus
        # -----------------------------
        # This combines the same information available to
        # the multi-agent system into one document.
        baseline_sections = [
            f"TITLE\n{title}",
            f"METADATA\n{metadata_text}",
            f"STORY / CONTEXT\n{story_text}",
        ]

        if has_materials:
            baseline_sections.append(
                f"RELATED MATERIALS\n{materials_text}"
            )

        baseline_text = "\n\n".join(baseline_sections)

        baseline_document = make_document(
            document_id=f"{record_id}::baseline",
            record_id=record_id,
            agent="single_agent_baseline",
            source_field="combined_story_materials_metadata",
            text=baseline_text,
            title=title,
            europeana_url=europeana_url,
        )

        baseline_documents.append(baseline_document)

        manifest_rows.append(
            {
                "record_id": record_id,
                "title": title,
                "included_in_final_corpus": "True",
                "reason": "Meets minimum narrative length",
                "story_words": str(word_count(story_text)),
                "materials_words": str(word_count(materials_text)),
            }
        )

    # --------------------------------------------------
    # Save files
    # --------------------------------------------------

    write_jsonl(STORY_OUTPUT, story_documents)
    write_jsonl(MATERIALS_OUTPUT, materials_documents)
    write_jsonl(METADATA_OUTPUT, metadata_documents)
    write_jsonl(BASELINE_OUTPUT, baseline_documents)
    write_manifest(MANIFEST_OUTPUT, manifest_rows)

    summary = {
        "input_candidate_records": len(records),
        "minimum_story_words": MIN_STORY_WORDS,
        "minimum_materials_words": MIN_MATERIALS_WORDS,
        "records_excluded_for_short_story": excluded_short_story,
        "final_unique_records": len(story_documents),
        "story_agent_documents": len(story_documents),
        "materials_agent_documents": len(materials_documents),
        "metadata_agent_documents": len(metadata_documents),
        "single_agent_baseline_documents": len(baseline_documents),
        "outputs": {
            "story_agent": str(STORY_OUTPUT),
            "materials_agent": str(MATERIALS_OUTPUT),
            "metadata_agent": str(METADATA_OUTPUT),
            "single_agent_baseline": str(BASELINE_OUTPUT),
            "manifest": str(MANIFEST_OUTPUT),
        },
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n--- Agent corpora prepared ---")
    print(f"Candidate records read: {len(records)}")
    print(f"Final records retained: {len(story_documents)}")
    print(f"Excluded for short story: {excluded_short_story}")

    print("\nDocuments created:")
    print(f"- Story agent: {len(story_documents)}")
    print(f"- Materials agent: {len(materials_documents)}")
    print(f"- Metadata agent: {len(metadata_documents)}")
    print(f"- Single-agent baseline: {len(baseline_documents)}")

    print("\nSaved to:")
    print(OUTPUT_DIR.resolve())

    print("\nSummary file:")
    print(SUMMARY_OUTPUT.resolve())


if __name__ == "__main__":
    main()