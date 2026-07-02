# scripts/prepare_agent_corpora.py

"""
Prepare validated Europeana agent corpora.

This script:
1. Starts from the original 220 retained records.
2. Excludes one manually verified exact duplicate.
3. Reads all English description segments for each record.
4. Assigns each segment to either:
   - narrative agent, or
   - materials agent
   using transparent content-based rules.
5. Creates a metadata agent corpus and a fair single-agent baseline.
6. Saves a full audit trail so every classification can be reviewed.

Important:
- This does not delete raw data.
- This does not alter the original agent_corpora folder.
- The new corpora are candidates pending final validation.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CANDIDATES_CSV = Path(
    "data/cleaned/ww1_exploration_english_candidates.csv"
)

# This is the manifest from the first corpus-preparation run.
# It contains the original 220 retained records.
ORIGINAL_MANIFEST_CSV = Path(
    "data/processed/agent_corpora/corpus_manifest.csv"
)

# New output folder. Your original files remain untouched.
OUTPUT_DIR = Path(
    "data/processed/agent_corpora_validated"
)

NARRATIVE_OUTPUT = OUTPUT_DIR / "narrative_agent_documents.jsonl"
MATERIALS_OUTPUT = OUTPUT_DIR / "materials_agent_documents.jsonl"
METADATA_OUTPUT = OUTPUT_DIR / "metadata_agent_documents.jsonl"
BASELINE_OUTPUT = OUTPUT_DIR / "single_agent_documents.jsonl"

MANIFEST_OUTPUT = OUTPUT_DIR / "corpus_manifest.csv"
CLASSIFICATION_AUDIT_OUTPUT = (
    OUTPUT_DIR / "description_classification_audit.csv"
)
AMBIGUOUS_SEGMENTS_OUTPUT = (
    OUTPUT_DIR / "ambiguous_description_segments.csv"
)
SUMMARY_OUTPUT = OUTPUT_DIR / "corpus_summary.json"


# --------------------------------------------------
# One manually verified duplicate
# --------------------------------------------------

# Both records contain the same narrative about Wilhelm Große Munkenbeck.
# Keep the broader record concerning the leather case with photos/documents.
# Exclude the narrower aviator-helmet entry from experimental retrieval.
EXCLUDED_RECORDS = {
    "/2020601/https___1914_1918_europeana_eu_contributions_10818": (
        "Exact duplicate narrative of record "
        "https___1914_1918_europeana_eu_contributions_10794. "
        "The retained record has broader associated material."
    )
}


# --------------------------------------------------
# Classification rules
# --------------------------------------------------

# These are deliberately simple and inspectable.
# They are not a black-box classifier.

NARRATIVE_PATTERNS = [
    (r"\b(?:was|were) born\b", 3),
    (
        r"\b(?:was|were) "
        r"(?:mobilized|mobilised|drafted|called up|enlisted|"
        r"wounded|injured|killed|captured|taken prisoner|"
        r"evacuated|hospitalized|hospitalised|released|"
        r"demobilized|demobilised|decorated)\b",
        2,
    ),
    (
        r"\b(?:served|fought|joined|married|died|returned|"
        r"survived|worked|lived|moved|travelled|traveled|"
        r"recounted|described|experienced)\b",
        1,
    ),
    (
        r"\b(?:grandfather|grandmother|father|mother|wife|"
        r"husband|daughter|son|brother|sister|family)\b",
        1,
    ),
    (r"\b(?:front|trench|battle|campaign|captivity|prisoner)\b", 1),
]

MATERIALS_PATTERNS = [
    (
        r"\b(?:letter|letters|postcard|postcards|diary|diaries|"
        r"notebook|notebooks|photograph|photographs|photo|photos|"
        r"album|albums|medal|medals|document|documents|booklet|"
        r"booklets|card|cards|certificate|certificates|map|maps|"
        r"drawing|drawings|sketch|sketches|manuscript|manuscripts|"
        r"telegram|telegrams|newspaper|portrait|portraits|"
        r"watercolor|watercolours|watercolors|collection|"
        r"objects|object|artefact|artefacts|artifact|artifacts)\b",
        2,
    ),
    (
        r"\b(?:digitised|digitized|transcript|transcription|"
        r"selected extracts|full digitized version|"
        r"full digitised version)\b",
        2,
    ),
]

# A segment is called “ambiguous” when its two scores are close.
AMBIGUITY_MARGIN = 1


# --------------------------------------------------
# General helper functions
# --------------------------------------------------

def clean_text(value: Any) -> str:
    """Convert a value into clean, single-spaced text."""
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def word_count(text: str) -> int:
    """Count approximate words in a text string."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def split_pipe_values(value: str) -> list[str]:
    """Convert '1914 | 1918' into ['1914', '1918']."""
    text = clean_text(value)

    if not text:
        return []

    return [
        part.strip()
        for part in text.split("|")
        if part.strip()
    ]


def unique_preserving_order(values: list[str]) -> list[str]:
    """Remove duplicate text segments while preserving their order."""
    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        text = clean_text(value)

        if text and text not in seen:
            unique_values.append(text)
            seen.add(text)

    return unique_values


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write CSV, including a header even if no rows exist."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(
    path: Path,
    documents: list[dict[str, Any]],
) -> None:
    """Write one JSON object per line."""
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(
                json.dumps(document, ensure_ascii=False) + "\n"
            )


# --------------------------------------------------
# Load records
# --------------------------------------------------

def load_original_retained_ids() -> set[str]:
    """
    Read the original manifest to retain the same 220 records
    selected in the earlier preparation stage.
    """
    if not ORIGINAL_MANIFEST_CSV.exists():
        raise FileNotFoundError(
            "Could not find the original corpus manifest:\n"
            f"{ORIGINAL_MANIFEST_CSV.resolve()}\n\n"
            "Run the earlier corpus-preparation script first."
        )

    retained_ids: set[str] = set()

    with ORIGINAL_MANIFEST_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            record_id = clean_text(row.get("record_id"))
            included = clean_text(
                row.get("included_in_final_corpus")
            ).lower()

            if record_id and included == "true":
                retained_ids.add(record_id)

    if not retained_ids:
        raise RuntimeError(
            "The original manifest was found, but it contains "
            "no retained record IDs."
        )

    return retained_ids


def load_candidate_rows() -> list[dict[str, str]]:
    """Load candidate rows from the normalised Europeana CSV."""
    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(
            "Could not find the candidate CSV:\n"
            f"{CANDIDATES_CSV.resolve()}\n\n"
            "Run normalise_europeana.py first."
        )

    with CANDIDATES_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(
            "The candidate CSV exists but contains no records."
        )

    return rows


def load_working_records() -> list[dict[str, str]]:
    """
    Keep original retained records, then exclude the manually
    verified exact duplicate.
    """
    retained_ids = load_original_retained_ids()
    candidate_rows = load_candidate_rows()

    working_records: list[dict[str, str]] = []

    for row in candidate_rows:
        record_id = clean_text(row.get("record_id"))

        if record_id not in retained_ids:
            continue

        if record_id in EXCLUDED_RECORDS:
            continue

        working_records.append(row)

    if not working_records:
        raise RuntimeError(
            "No working records were found after filtering."
        )

    return working_records


# --------------------------------------------------
# Description extraction and classification
# --------------------------------------------------

def get_description_segments(
    row: dict[str, str],
) -> list[str]:
    """
    Reconstruct the individual English description values.

    english_text_all preserves the original descriptions separated
    by blank lines. This is better than assuming the longest one
    is always the narrative.
    """
    english_text_all = str(row.get("english_text_all", "")).strip()

    if english_text_all:
        segments = [
            clean_text(part)
            for part in re.split(r"\n\s*\n", english_text_all)
            if clean_text(part)
        ]

        segments = unique_preserving_order(segments)

        if segments:
            return segments

    # Fallback for unexpected CSV formatting.
    fallback_segments = [
        clean_text(row.get("english_longest_description"))
    ]

    other_descriptions = clean_text(
        row.get("english_other_descriptions")
    )

    if other_descriptions:
        fallback_segments.extend(
            clean_text(part)
            for part in other_descriptions.split(" | ")
            if clean_text(part)
        )

    return unique_preserving_order(fallback_segments)


def score_text(
    text: str,
    patterns: list[tuple[str, int]],
) -> tuple[int, int]:
    """
    Return:
    - weighted score
    - number of matched pattern occurrences
    """
    score = 0
    matches = 0

    for pattern, weight in patterns:
        occurrences = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        score += len(occurrences) * weight
        matches += len(occurrences)

    return score, matches


def count_list_markers(text: str) -> int:
    """
    Detect numbered or bullet-style material inventories.

    Examples:
    - '1. Photograph'
    - '- Diary'
    - '03 Letter'
    """
    numbered = re.findall(
        r"(?:^|[.;])\s*(?:\d{1,3}[-.)]|[-•])\s*",
        text,
    )

    return len(numbered)


def classify_segment(text: str) -> dict[str, Any]:
    """
    Assign a text segment to either narrative or materials.

    The decision and scores are saved to the audit CSV.
    """
    narrative_score, narrative_matches = score_text(
        text,
        NARRATIVE_PATTERNS,
    )

    materials_score, materials_matches = score_text(
        text,
        MATERIALS_PATTERNS,
    )

    list_markers = count_list_markers(text)

    # Numbered inventories are strong evidence of a materials list.
    materials_score += min(list_markers, 3) * 2

    score_difference = narrative_score - materials_score
    is_ambiguous = abs(score_difference) <= AMBIGUITY_MARGIN

    if materials_score > narrative_score:
        assigned_agent = "materials"
    else:
        assigned_agent = "narrative"

    if is_ambiguous:
        reason = (
            "Scores are close, so this segment should be treated "
            "as a manual-review candidate."
        )
    elif assigned_agent == "narrative":
        reason = (
            "Narrative indicators outweigh associated-material "
            "indicators."
        )
    else:
        reason = (
            "Associated-material indicators outweigh narrative "
            "indicators."
        )

    return {
        "assigned_agent": assigned_agent,
        "narrative_score": narrative_score,
        "materials_score": materials_score,
        "narrative_matches": narrative_matches,
        "materials_matches": materials_matches,
        "list_markers": list_markers,
        "is_ambiguous": is_ambiguous,
        "reason": reason,
    }


# --------------------------------------------------
# Document construction
# --------------------------------------------------

def make_metadata_text(row: dict[str, str]) -> str:
    """Create a clean structured metadata representation."""
    title = clean_text(row.get("title_variants"))
    record_type = clean_text(row.get("record_type"))
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
        split_pipe_values(row.get("original_item_language", ""))
    )
    provider = clean_text(row.get("data_provider"))
    rights = clean_text(row.get("rights"))

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

    return "\n".join(
        line
        for line in metadata_lines
        if not line.endswith(": ")
    )


def make_document(
    *,
    document_id: str,
    record_id: str,
    agent: str,
    source_field: str,
    title: str,
    text: str,
    europeana_url: str,
    segment_count: int,
) -> dict[str, Any]:
    """Build one standard JSONL document."""
    return {
        "document_id": document_id,
        "record_id": record_id,
        "agent": agent,
        "source_field": source_field,
        "title": title,
        "text": text,
        "source_segment_count": segment_count,
        "europeana_url": europeana_url,
    }


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading retained Europeana records...")
    records = load_working_records()

    narrative_documents: list[dict[str, Any]] = []
    materials_documents: list[dict[str, Any]] = []
    metadata_documents: list[dict[str, Any]] = []
    baseline_documents: list[dict[str, Any]] = []

    manifest_rows: list[dict[str, str]] = []
    classification_rows: list[dict[str, str]] = []
    ambiguous_rows: list[dict[str, str]] = []

    total_segments = 0
    ambiguous_segment_count = 0
    records_with_narrative = 0
    records_with_materials = 0

    for row in records:
        record_id = clean_text(row.get("record_id"))
        title = clean_text(row.get("title_variants"))
        europeana_url = clean_text(row.get("europeana_url"))

        segments = get_description_segments(row)

        if not segments:
            raise RuntimeError(
                "No English description segments found for:\n"
                f"{record_id}"
            )

        total_segments += len(segments)

        narrative_segments: list[str] = []
        materials_segments: list[str] = []
        record_ambiguous_segments = 0

        for segment_index, segment_text in enumerate(
            segments,
            start=1,
        ):
            classification = classify_segment(segment_text)

            assigned_agent = classification["assigned_agent"]

            if assigned_agent == "narrative":
                narrative_segments.append(segment_text)
            else:
                materials_segments.append(segment_text)

            if classification["is_ambiguous"]:
                record_ambiguous_segments += 1
                ambiguous_segment_count += 1

            audit_row = {
                "record_id": record_id,
                "title": title,
                "segment_index": str(segment_index),
                "segment_word_count": str(word_count(segment_text)),
                "assigned_agent": assigned_agent,
                "narrative_score": str(
                    classification["narrative_score"]
                ),
                "materials_score": str(
                    classification["materials_score"]
                ),
                "narrative_matches": str(
                    classification["narrative_matches"]
                ),
                "materials_matches": str(
                    classification["materials_matches"]
                ),
                "list_markers": str(
                    classification["list_markers"]
                ),
                "is_ambiguous": str(
                    classification["is_ambiguous"]
                ),
                "classification_reason": str(
                    classification["reason"]
                ),
                "text": segment_text,
            }

            classification_rows.append(audit_row)

            if classification["is_ambiguous"]:
                ambiguous_rows.append(audit_row)

        metadata_text = make_metadata_text(row)

        # All English descriptive text appears exactly once across
        # narrative and materials agent corpora.
        all_description_text = "\n\n".join(segments)
        narrative_text = "\n\n".join(narrative_segments)
        materials_text = "\n\n".join(materials_segments)

        if narrative_text:
            records_with_narrative += 1

            narrative_documents.append(
                make_document(
                    document_id=f"{record_id}::narrative",
                    record_id=record_id,
                    agent="narrative",
                    source_field=(
                        "dcDescriptionLangAware.en "
                        "classified_as_narrative"
                    ),
                    title=title,
                    text=narrative_text,
                    europeana_url=europeana_url,
                    segment_count=len(narrative_segments),
                )
            )

        if materials_text:
            records_with_materials += 1

            materials_documents.append(
                make_document(
                    document_id=f"{record_id}::materials",
                    record_id=record_id,
                    agent="materials",
                    source_field=(
                        "dcDescriptionLangAware.en "
                        "classified_as_materials"
                    ),
                    title=title,
                    text=materials_text,
                    europeana_url=europeana_url,
                    segment_count=len(materials_segments),
                )
            )

        metadata_documents.append(
            make_document(
                document_id=f"{record_id}::metadata",
                record_id=record_id,
                agent="metadata",
                source_field="structured_metadata",
                title=title,
                text=metadata_text,
                europeana_url=europeana_url,
                segment_count=1,
            )
        )

        # Baseline receives the same metadata and all descriptive text.
        baseline_text = "\n\n".join(
            [
                f"TITLE\n{title}",
                f"STRUCTURED METADATA\n{metadata_text}",
                f"DESCRIPTIVE TEXT\n{all_description_text}",
            ]
        )

        baseline_documents.append(
            make_document(
                document_id=f"{record_id}::baseline",
                record_id=record_id,
                agent="single_agent_baseline",
                source_field=(
                    "structured_metadata_and_all_english_descriptions"
                ),
                title=title,
                text=baseline_text,
                europeana_url=europeana_url,
                segment_count=len(segments),
            )
        )

        manifest_rows.append(
            {
                "record_id": record_id,
                "title": title,
                "included_in_validated_candidate_corpus": "True",
                "exclusion_reason": "",
                "description_segments_total": str(len(segments)),
                "narrative_segments": str(len(narrative_segments)),
                "materials_segments": str(len(materials_segments)),
                "ambiguous_segments": str(record_ambiguous_segments),
                "has_narrative_document": str(bool(narrative_text)),
                "has_materials_document": str(bool(materials_text)),
                "baseline_description_words": str(
                    word_count(all_description_text)
                ),
            }
        )

    # Record the excluded duplicate in the manifest too.
    candidate_rows = load_candidate_rows()

    for row in candidate_rows:
        record_id = clean_text(row.get("record_id"))

        if record_id not in EXCLUDED_RECORDS:
            continue

        manifest_rows.append(
            {
                "record_id": record_id,
                "title": clean_text(row.get("title_variants")),
                "included_in_validated_candidate_corpus": "False",
                "exclusion_reason": EXCLUDED_RECORDS[record_id],
                "description_segments_total": "",
                "narrative_segments": "",
                "materials_segments": "",
                "ambiguous_segments": "",
                "has_narrative_document": "False",
                "has_materials_document": "False",
                "baseline_description_words": "",
            }
        )

    # --------------------------------------------------
    # Save all outputs
    # --------------------------------------------------

    write_jsonl(NARRATIVE_OUTPUT, narrative_documents)
    write_jsonl(MATERIALS_OUTPUT, materials_documents)
    write_jsonl(METADATA_OUTPUT, metadata_documents)
    write_jsonl(BASELINE_OUTPUT, baseline_documents)

    write_csv(
        MANIFEST_OUTPUT,
        manifest_rows,
        fieldnames=[
            "record_id",
            "title",
            "included_in_validated_candidate_corpus",
            "exclusion_reason",
            "description_segments_total",
            "narrative_segments",
            "materials_segments",
            "ambiguous_segments",
            "has_narrative_document",
            "has_materials_document",
            "baseline_description_words",
        ],
    )

    write_csv(
        CLASSIFICATION_AUDIT_OUTPUT,
        classification_rows,
        fieldnames=[
            "record_id",
            "title",
            "segment_index",
            "segment_word_count",
            "assigned_agent",
            "narrative_score",
            "materials_score",
            "narrative_matches",
            "materials_matches",
            "list_markers",
            "is_ambiguous",
            "classification_reason",
            "text",
        ],
    )

    write_csv(
        AMBIGUOUS_SEGMENTS_OUTPUT,
        ambiguous_rows,
        fieldnames=[
            "record_id",
            "title",
            "segment_index",
            "segment_word_count",
            "assigned_agent",
            "narrative_score",
            "materials_score",
            "narrative_matches",
            "materials_matches",
            "list_markers",
            "is_ambiguous",
            "classification_reason",
            "text",
        ],
    )

    summary = {
        "corpus_status": (
            "validated_candidate_pending_final_manual_review"
        ),
        "original_retained_records": 220,
        "verified_duplicate_records_excluded": len(EXCLUDED_RECORDS),
        "validated_candidate_records": len(baseline_documents),
        "total_english_description_segments": total_segments,
        "ambiguous_description_segments": ambiguous_segment_count,
        "narrative_agent_documents": len(narrative_documents),
        "materials_agent_documents": len(materials_documents),
        "metadata_agent_documents": len(metadata_documents),
        "single_agent_baseline_documents": len(baseline_documents),
        "records_with_narrative_text": records_with_narrative,
        "records_with_materials_text": records_with_materials,
        "agent_assignment_method": (
            "Transparent keyword-and-list-marker scoring of "
            "individual English description segments."
        ),
        "important_note": (
            "The classification audit should be inspected before "
            "the three-agent design is treated as final."
        ),
        "outputs": {
            "narrative_agent": str(NARRATIVE_OUTPUT),
            "materials_agent": str(MATERIALS_OUTPUT),
            "metadata_agent": str(METADATA_OUTPUT),
            "single_agent_baseline": str(BASELINE_OUTPUT),
            "manifest": str(MANIFEST_OUTPUT),
            "classification_audit": str(
                CLASSIFICATION_AUDIT_OUTPUT
            ),
            "ambiguous_segments": str(
                AMBIGUOUS_SEGMENTS_OUTPUT
            ),
        },
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Terminal output
    # --------------------------------------------------

    print("\n--- Validated candidate corpora prepared ---")
    print(f"Original retained records: 220")
    print(
        "Verified duplicate records excluded: "
        f"{len(EXCLUDED_RECORDS)}"
    )
    print(
        "Validated candidate records: "
        f"{len(baseline_documents)}"
    )
    print(f"English description segments classified: {total_segments}")
    print(
        "Ambiguous segments flagged for audit: "
        f"{ambiguous_segment_count}"
    )

    print("\nDocuments created:")
    print(f"- Narrative agent: {len(narrative_documents)}")
    print(f"- Materials agent: {len(materials_documents)}")
    print(f"- Metadata agent: {len(metadata_documents)}")
    print(f"- Single-agent baseline: {len(baseline_documents)}")

    print("\nSaved to:")
    print(OUTPUT_DIR.resolve())

    print("\nKey files:")
    print(f"- {SUMMARY_OUTPUT.resolve()}")
    print(f"- {AMBIGUOUS_SEGMENTS_OUTPUT.resolve()}")
    print(f"- {CLASSIFICATION_AUDIT_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()