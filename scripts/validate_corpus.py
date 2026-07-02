# scripts/validate_corpus.py

"""
Validate the current 220-record Europeana working corpus.

This script does NOT change, remove, or overwrite the corpus.
It creates an audit report, a small manual-review sample, and
a list of possible duplicate records for inspection.
"""

import csv
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CANDIDATES_CSV = Path(
    "data/cleaned/ww1_exploration_english_candidates.csv"
)

MANIFEST_CSV = Path(
    "data/processed/agent_corpora/corpus_manifest.csv"
)

PROFILE_JSON = Path(
    "data/cleaned/ww1_exploration_profile.json"
)

OUTPUT_DIR = Path("data/validation")

SUMMARY_OUTPUT = OUTPUT_DIR / "corpus_validation_summary.json"
REPORT_OUTPUT = OUTPUT_DIR / "corpus_validation_report.md"
SAMPLE_OUTPUT = OUTPUT_DIR / "manual_review_sample.csv"
DUPLICATES_OUTPUT = OUTPUT_DIR / "possible_duplicate_pairs.csv"

# These are audit thresholds only.
# They do not automatically remove any records.
MEANINGFUL_MATERIALS_WORDS = 20
POSSIBLE_DUPLICATE_JACCARD_THRESHOLD = 0.80

RANDOM_SEED = 20260628


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


def normalise_text(text: str) -> str:
    """Normalise text for duplicate comparison."""
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    """
    Convert text into a set of meaningful tokens.

    Tokens shorter than three characters are ignored because they
    create too much accidental overlap.
    """
    return set(re.findall(r"\b[a-z0-9]{3,}\b", normalise_text(text)))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Calculate token-set similarity between two strings."""
    tokens_a = token_set(text_a)
    tokens_b = token_set(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def percentile(values: list[int], percentage: float) -> float:
    """Return a simple interpolated percentile."""
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    index = (len(ordered) - 1) * percentage
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def numeric_summary(values: list[int]) -> dict[str, float | int]:
    """Create readable summary statistics."""
    if not values:
        return {
            "count": 0,
            "minimum": 0,
            "p25": 0,
            "median": 0,
            "p75": 0,
            "maximum": 0,
            "mean": 0,
        }

    return {
        "count": len(values),
        "minimum": min(values),
        "p25": round(percentile(values, 0.25), 2),
        "median": round(statistics.median(values), 2),
        "p75": round(percentile(values, 0.75), 2),
        "maximum": max(values),
        "mean": round(statistics.mean(values), 2),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write rows to CSV even when no rows are present."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# Load and prepare records
# --------------------------------------------------

def load_final_record_ids() -> set[str]:
    """
    Read the manifest and return only records retained by the
    previous corpus-preparation script.
    """
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            "Could not find the corpus manifest:\n"
            f"{MANIFEST_CSV.resolve()}\n\n"
            "Run prepare_agent_corpora.py first."
        )

    retained_ids: set[str] = set()

    with MANIFEST_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            record_id = clean_text(row.get("record_id"))
            included = clean_text(
                row.get("included_in_final_corpus")
            )

            if record_id and included.lower() == "true":
                retained_ids.add(record_id)

    if not retained_ids:
        raise RuntimeError(
            "The manifest was found, but no retained record IDs "
            "were recorded."
        )

    return retained_ids


def load_final_records() -> list[dict[str, str]]:
    """Load only the 220 retained records from the candidate CSV."""
    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(
            "Could not find the candidate CSV:\n"
            f"{CANDIDATES_CSV.resolve()}\n\n"
            "Run normalise_europeana.py first."
        )

    retained_ids = load_final_record_ids()

    with CANDIDATES_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        candidate_rows = list(reader)

    final_rows = [
        row
        for row in candidate_rows
        if clean_text(row.get("record_id")) in retained_ids
    ]

    if not final_rows:
        raise RuntimeError(
            "No final records were found after matching the "
            "candidate CSV to the corpus manifest."
        )

    return final_rows


# --------------------------------------------------
# Corpus audit
# --------------------------------------------------

def make_audit_rows(
    records: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Add audit fields without changing original record content."""
    audit_rows: list[dict[str, str]] = []

    for row in records:
        story_text = clean_text(
            row.get("english_longest_description")
        )
        materials_text = clean_text(
            row.get("english_other_descriptions")
        )

        audit_rows.append(
            {
                "record_id": clean_text(row.get("record_id")),
                "title": clean_text(row.get("title_variants")),
                "original_item_language": clean_text(
                    row.get("original_item_language")
                ),
                "rights": clean_text(row.get("rights")),
                "europeana_url": clean_text(
                    row.get("europeana_url")
                ),
                "story_text": story_text,
                "materials_text": materials_text,
                "story_words": str(word_count(story_text)),
                "materials_words": str(word_count(materials_text)),
                "has_meaningful_materials": str(
                    word_count(materials_text)
                    >= MEANINGFUL_MATERIALS_WORDS
                ),
            }
        )

    return audit_rows


def find_possible_duplicates(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Flag highly similar stories for manual inspection.

    These are only possible duplicates. Nothing is removed.
    """
    possible_duplicates: list[dict[str, str]] = []

    for row_a, row_b in combinations(rows, 2):
        story_a = row_a["story_text"]
        story_b = row_b["story_text"]

        # Avoid comparing extremely tiny texts.
        if min(word_count(story_a), word_count(story_b)) < 40:
            continue

        similarity = jaccard_similarity(story_a, story_b)

        if similarity >= POSSIBLE_DUPLICATE_JACCARD_THRESHOLD:
            possible_duplicates.append(
                {
                    "record_id_a": row_a["record_id"],
                    "title_a": row_a["title"],
                    "record_id_b": row_b["record_id"],
                    "title_b": row_b["title"],
                    "story_jaccard_similarity": str(
                        round(similarity, 3)
                    ),
                }
            )

    return sorted(
        possible_duplicates,
        key=lambda row: float(
            row["story_jaccard_similarity"]
        ),
        reverse=True,
    )


def select_manual_review_sample(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Select a reproducible 12-record sample across short, medium,
    and long narratives.

    Four records are selected from each band where possible.
    """
    random_generator = random.Random(RANDOM_SEED)

    strata = {
        "short_story_40_to_99_words": [],
        "medium_story_100_to_199_words": [],
        "long_story_200_plus_words": [],
    }

    for row in rows:
        story_words = int(row["story_words"])

        if 40 <= story_words <= 99:
            strata["short_story_40_to_99_words"].append(row)
        elif 100 <= story_words <= 199:
            strata["medium_story_100_to_199_words"].append(row)
        else:
            strata["long_story_200_plus_words"].append(row)

    sample_rows: list[dict[str, str]] = []
    sample_number = 1

    for stratum_name, stratum_rows in strata.items():
        selected = random_generator.sample(
            stratum_rows,
            k=min(4, len(stratum_rows)),
        )

        for row in selected:
            sample_rows.append(
                {
                    "sample_id": f"S{sample_number:02d}",
                    "stratum": stratum_name,
                    "record_id": row["record_id"],
                    "title": row["title"],
                    "original_item_language": (
                        row["original_item_language"]
                    ),
                    "story_words": row["story_words"],
                    "materials_words": row["materials_words"],
                    "story_text": row["story_text"],
                    "materials_text": row["materials_text"],
                    "review_answerable_yes_no": "",
                    "review_english_coherent_yes_no": "",
                    "review_materials_distinct_yes_no": "",
                    "review_red_flag_yes_no": "",
                    "review_notes": "",
                }
            )

            sample_number += 1

    return sample_rows


# --------------------------------------------------
# Report generation
# --------------------------------------------------

def load_exploration_profile() -> dict[str, Any]:
    """Read previous exploration statistics when available."""
    if not PROFILE_JSON.exists():
        return {}

    try:
        return json.loads(
            PROFILE_JSON.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        return {}


def make_markdown_report(
    summary: dict[str, Any],
) -> str:
    """Create a readable validation report."""
    selection = summary["selection"]
    story = summary["story_text"]
    materials = summary["materials_text"]
    integrity = summary["integrity"]

    return f"""# Europeana Corpus Validation Report

## Purpose

This report validates the current working corpus before it is fixed
for comparative single-agent and multi-agent RAG evaluation.

This audit does not alter the corpus. It reports corpus composition,
text coverage, possible duplicates, and a reproducible manual-review
sample.

## Corpus selection

- Raw exploratory records: {selection["raw_exploratory_records"]}
- Records with English descriptive metadata: {selection["records_with_english_description"]}
- Candidate records before short-story removal: {selection["candidate_records"]}
- Records retained in the current working corpus: {selection["final_working_corpus_records"]}
- Selection rule used for the retained corpus: English descriptive metadata, rights information, and at least 40 words in the main narrative field.

## Main story text

- Minimum story length: {story["minimum"]} words
- 25th percentile: {story["p25"]} words
- Median: {story["median"]} words
- 75th percentile: {story["p75"]} words
- Maximum: {story["maximum"]} words

## Materials text

- Records with any materials text: {materials["records_with_any_materials"]}
- Records with at least 5 materials words: {materials["records_with_5_plus_words"]}
- Records with at least {MEANINGFUL_MATERIALS_WORDS} materials words: {materials["records_with_meaningful_materials"]}
- Median materials length, where present: {materials["summary_non_empty"]["median"]} words

The materials field requires manual review because some records may
contain detailed descriptions of diaries, letters, photographs, medals
or albums, while others may contain only short inventory labels.

## Integrity checks

- Missing title: {integrity["missing_title"]}
- Missing story text: {integrity["missing_story_text"]}
- Missing rights field: {integrity["missing_rights"]}
- Missing Europeana URL: {integrity["missing_europeana_url"]}
- Possible high-similarity duplicate pairs: {integrity["possible_duplicate_pairs"]}

## Manual validation sample

A fixed random sample of 12 records was created across short,
medium and long story-text bands. It should be reviewed for:

1. Whether the record can support evidence-based questions.
2. Whether the English text is sufficiently coherent.
3. Whether the materials description contributes distinct information.
4. Whether there are any obvious factual, translation or metadata red flags.

The sample is saved as:

`data/validation/manual_review_sample.csv`

## Decision status

The corpus should only be treated as final after:

1. Reviewing the 12-record sample.
2. Inspecting any possible duplicate pairs.
3. Deciding whether the materials field supports a standalone specialist agent.
"""


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading the retained corpus...")
    records = load_final_records()
    audit_rows = make_audit_rows(records)

    story_lengths = [
        int(row["story_words"])
        for row in audit_rows
    ]

    materials_lengths_non_empty = [
        int(row["materials_words"])
        for row in audit_rows
        if int(row["materials_words"]) > 0
    ]

    records_with_any_materials = sum(
        int(row["materials_words"]) > 0
        for row in audit_rows
    )

    records_with_5_plus_materials = sum(
        int(row["materials_words"]) >= 5
        for row in audit_rows
    )

    records_with_meaningful_materials = sum(
        int(row["materials_words"])
        >= MEANINGFUL_MATERIALS_WORDS
        for row in audit_rows
    )

    print("Checking possible duplicate records...")
    possible_duplicates = find_possible_duplicates(audit_rows)

    print("Creating the manual-review sample...")
    manual_review_sample = select_manual_review_sample(audit_rows)

    exploration_profile = load_exploration_profile()

    summary = {
        "selection": {
            "raw_exploratory_records": (
                exploration_profile.get("total_unique_records", "Unknown")
            ),
            "records_with_english_description": (
                exploration_profile.get(
                    "records_with_english_description",
                    "Unknown",
                )
            ),
            "candidate_records": (
                exploration_profile.get(
                    "records_recommended_for_manual_review",
                    "Unknown",
                )
            ),
            "final_working_corpus_records": len(audit_rows),
            "selection_rule": (
                "English descriptive metadata, rights information, "
                "and at least 40 words in the main narrative field."
            ),
        },
        "story_text": numeric_summary(story_lengths),
        "materials_text": {
            "records_with_any_materials": (
                records_with_any_materials
            ),
            "records_with_5_plus_words": (
                records_with_5_plus_materials
            ),
            "records_with_meaningful_materials": (
                records_with_meaningful_materials
            ),
            "summary_non_empty": numeric_summary(
                materials_lengths_non_empty
            ),
        },
        "integrity": {
            "missing_title": sum(
                not row["title"]
                for row in audit_rows
            ),
            "missing_story_text": sum(
                not row["story_text"]
                for row in audit_rows
            ),
            "missing_rights": sum(
                not row["rights"]
                for row in audit_rows
            ),
            "missing_europeana_url": sum(
                not row["europeana_url"]
                for row in audit_rows
            ),
            "possible_duplicate_pairs": len(
                possible_duplicates
            ),
        },
        "manual_review": {
            "sample_size": len(manual_review_sample),
            "random_seed": RANDOM_SEED,
            "stratification": (
                "4 short, 4 medium and 4 long narratives "
                "where available."
            ),
        },
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    REPORT_OUTPUT.write_text(
        make_markdown_report(summary),
        encoding="utf-8",
    )

    write_csv(
        SAMPLE_OUTPUT,
        manual_review_sample,
        fieldnames=[
            "sample_id",
            "stratum",
            "record_id",
            "title",
            "original_item_language",
            "story_words",
            "materials_words",
            "story_text",
            "materials_text",
            "review_answerable_yes_no",
            "review_english_coherent_yes_no",
            "review_materials_distinct_yes_no",
            "review_red_flag_yes_no",
            "review_notes",
        ],
    )

    write_csv(
        DUPLICATES_OUTPUT,
        possible_duplicates,
        fieldnames=[
            "record_id_a",
            "title_a",
            "record_id_b",
            "title_b",
            "story_jaccard_similarity",
        ],
    )

    print("\n--- Corpus validation files created ---")
    print(f"Records audited: {len(audit_rows)}")
    print(
        "Records with meaningful materials text "
        f"({MEANINGFUL_MATERIALS_WORDS}+ words): "
        f"{records_with_meaningful_materials}"
    )
    print(
        "Possible duplicate pairs flagged: "
        f"{len(possible_duplicates)}"
    )
    print(
        "Manual-review sample size: "
        f"{len(manual_review_sample)}"
    )

    print("\nCreated:")
    print(f"- {SUMMARY_OUTPUT.resolve()}")
    print(f"- {REPORT_OUTPUT.resolve()}")
    print(f"- {SAMPLE_OUTPUT.resolve()}")
    print(f"- {DUPLICATES_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()