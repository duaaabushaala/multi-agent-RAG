# scripts/normalise_europeana.py
# Turned raw Europeana JSON into clean rows/fields and produced the 230 candidate records

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# This is the JSON list created by explore_europeana.py
INPUT_FILE = Path(
    "data/raw/europeana/exploration_sample/"
    "europeana_1914_1918_search_results.json"
)

OUTPUT_DIR = Path("data/cleaned")

ALL_RECORDS_CSV = OUTPUT_DIR / "ww1_exploration_normalised_all.csv"
ENGLISH_CANDIDATES_CSV = OUTPUT_DIR / "ww1_exploration_english_candidates.csv"
PROFILE_JSON = OUTPUT_DIR / "ww1_exploration_profile.json"

# A record needs at least 40 English metadata words before we consider it worth manually inspecting.
MIN_ENGLISH_WORDS_FOR_CANDIDATE = 40


# --------------------------------------------------
# Text helper functions
# --------------------------------------------------

def clean_text(value: Any) -> str:
    """
    Convert a value into clean, single-spaced text.

    Europeana fields may be strings, lists, dictionaries, or null.
    """
    if value is None:
        return ""

    if isinstance(value, list):
        text = " ".join(clean_text(item) for item in value)
        return re.sub(r"\s+", " ", text).strip()

    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
        return re.sub(r"\s+", " ", text).strip()

    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def text_list(value: Any) -> list[str]:
    """
    Return a cleaned list of non-empty text values.

    """
    if value is None:
        return []

    if isinstance(value, list):
        cleaned_values = []

        for item in value:
            text = clean_text(item)

            if text:
                cleaned_values.append(text)

        return cleaned_values

    text = clean_text(value)

    return [text] if text else []


def word_count(text: str) -> int:
    """Count words in a piece of text."""
    return len(re.findall(r"\b[\w'-]+\b", text))


# --------------------------------------------------
# Europeana field extraction functions
# --------------------------------------------------

def get_language_values(
    record: dict[str, Any],
    field_name: str,
    language_code: str,
) -> list[str]:
    """
    Extract language-aware text values from a Europeana field.

    Example:
    record["dcDescriptionLangAware"]["en"]
    """
    field = record.get(field_name, {})

    if not isinstance(field, dict):
        return []

    return text_list(field.get(language_code))


def get_label_values(value: Any) -> list[str]:
    """
    Extract labels from Europeana fields such as edmPlaceLabel.

    These are often structured like:
    [
        {"def": "Jerusalem"},
        {"def": "France"}
    ]
    """
    labels: list[str] = []

    if not isinstance(value, list):
        return labels

    for item in value:
        if isinstance(item, dict):
            label = clean_text(item.get("def"))

            if label:
                labels.append(label)

        else:
            label = clean_text(item)

            if label:
                labels.append(label)

    return labels


def get_title_variants(record: dict[str, Any]) -> list[str]:
    """
    Return all available title values.

    We keep all variants because some records have an English
    title in the normal 'title' field even when dcTitleLangAware
    only contains the original-language title.
    """
    return text_list(record.get("title"))


def get_english_subjects(record: dict[str, Any]) -> list[str]:
    """
    Extract English concept labels, for example:
    World War I, Multiple, etc.
    """
    return get_language_values(
        record,
        "edmConceptPrefLabelLangAware",
        "en",
    )


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_all_records() -> list[dict[str, Any]]:
    """
    Read the JSON list created by explore_europeana.py.

    The exploration file should contain a list of Europeana records.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Exploration JSON file not found:\n"
            f"{INPUT_FILE.resolve()}\n\n"
            "Check that explore_europeana.py created the file, "
            "and confirm the filename matches this script."
        )

    try:
        payload = json.loads(
            INPUT_FILE.read_text(encoding="utf-8")
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Could not read valid JSON from {INPUT_FILE.name}: {error}"
        ) from error

    if not isinstance(payload, list):
        raise RuntimeError(
            "Expected the exploration JSON file to contain a list of records."
        )

    records = [
        record
        for record in payload
        if isinstance(record, dict)
    ]

    if not records:
        raise RuntimeError(
            "The exploration JSON file was found, but it contains no usable records."
        )

    # Record where each row originally came from.
    for record in records:
        record["_source_file"] = INPUT_FILE.name

    return records


# --------------------------------------------------
# Normalisation
# --------------------------------------------------

def normalise_record(record: dict[str, Any]) -> dict[str, str]:
    """
    Convert one nested Europeana record into one flat CSV row.
    """
    record_id = clean_text(record.get("id"))

    # Extract only the English translated metadata descriptions.
    english_descriptions = get_language_values(
        record,
        "dcDescriptionLangAware",
        "en",
    )

    # The longest English description is usually the main story/narrative.
    english_longest_description = ""
    english_other_descriptions: list[str] = []

    if english_descriptions:
        english_longest_description = max(
            english_descriptions,
            key=word_count,
        )

        english_other_descriptions = [
            description
            for description in english_descriptions
            if description != english_longest_description
        ]

    # Preserve every English description, separated clearly.
    english_text_all = "\n\n".join(english_descriptions)

    title_variants = get_title_variants(record)
    original_languages = text_list(record.get("dcLanguage"))
    year_values = text_list(record.get("year"))
    rights_values = text_list(record.get("rights"))
    place_values = get_label_values(record.get("edmPlaceLabel"))
    english_subjects = get_english_subjects(record)

    english_words = word_count(english_text_all)

    # This identifies records that have a stable rights statement
    # and enough English descriptive metadata to inspect manually.
    recommended_for_manual_review = (
        english_words >= MIN_ENGLISH_WORDS_FOR_CANDIDATE
        and bool(rights_values)
    )

    # Attachments are likely to have IDs ending in _attachments_...
    is_attachment = "_attachments_" in record_id

    return {
        "record_id": record_id,
        "source_file": clean_text(record.get("_source_file")),
        "record_role": (
            "attachment"
            if is_attachment
            else "contribution_or_parent_record"
        ),
        "title_variants": " | ".join(title_variants),
        "data_provider": " | ".join(
            text_list(record.get("dataProvider"))
        ),
        "provider": " | ".join(
            text_list(record.get("provider"))
        ),
        "collection_name": " | ".join(
            text_list(record.get("europeanaCollectionName"))
        ),
        "record_type": clean_text(record.get("type")),
        "original_item_language": " | ".join(original_languages),
        "year_values": " | ".join(year_values),
        "place_values": " | ".join(place_values),
        "english_subjects": " | ".join(english_subjects),
        "rights": " | ".join(rights_values),
        "english_description_count": str(len(english_descriptions)),
        "english_description_word_count": str(english_words),
        "english_longest_description": english_longest_description,
        "english_other_descriptions": " | ".join(
            english_other_descriptions
        ),
        "english_text_all": english_text_all,
        "has_english_description": str(bool(english_descriptions)),
        "recommended_for_manual_review": str(
            recommended_for_manual_review
        ),
        "europeana_url": (
            f"https://www.europeana.eu/en/item{record_id}"
            if record_id
            else ""
        ),
    }


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a list of row dictionaries to a CSV file."""
    if not rows:
        raise RuntimeError(
            f"No rows were available to write to {path.name}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main() -> None:
    print("Reading exploration JSON from:")
    print(INPUT_FILE.resolve())

    raw_records = load_all_records()

    print(f"\nRaw records loaded: {len(raw_records)}")

    rows = [
        normalise_record(record)
        for record in raw_records
    ]

    # Remove duplicate IDs, preserving the first occurrence.
    unique_rows: dict[str, dict[str, str]] = {}

    for row in rows:
        record_id = row["record_id"]

        if record_id and record_id not in unique_rows:
            unique_rows[record_id] = row

    rows = list(unique_rows.values())

    # Keep only records with enough English metadata and rights information.
    english_candidates = [
        row
        for row in rows
        if row["recommended_for_manual_review"] == "True"
    ]

    write_csv(ALL_RECORDS_CSV, rows)
    write_csv(ENGLISH_CANDIDATES_CSV, english_candidates)

    english_word_counts = [
        int(row["english_description_word_count"])
        for row in rows
    ]

    records_with_english = sum(
        row["has_english_description"] == "True"
        for row in rows
    )

    role_counts = Counter(
        row["record_role"]
        for row in rows
    )

    original_language_counts = Counter(
        row["original_item_language"] or "Unknown"
        for row in rows
    )

    profile = {
        "input_file": str(INPUT_FILE),
        "total_records_loaded": len(raw_records),
        "total_unique_records": len(rows),
        "records_with_english_description": records_with_english,
        "english_description_coverage_percent": round(
            (records_with_english / len(rows)) * 100,
            2,
        ) if rows else 0,
        "minimum_candidate_word_threshold": (
            MIN_ENGLISH_WORDS_FOR_CANDIDATE
        ),
        "records_recommended_for_manual_review": len(
            english_candidates
        ),
        "median_english_description_words": round(
            statistics.median(english_word_counts),
            2,
        ) if english_word_counts else 0,
        "record_role_counts": dict(role_counts),
        "most_common_original_item_languages": dict(
            original_language_counts.most_common(15)
        ),
    }

    PROFILE_JSON.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n--- Normalisation complete ---")
    print(f"Unique records processed: {len(rows)}")
    print(
        "Records with English description metadata: "
        f"{records_with_english}/{len(rows)} "
        f"({profile['english_description_coverage_percent']}%)"
    )
    print(
        "Records recommended for manual review "
        f"({MIN_ENGLISH_WORDS_FOR_CANDIDATE}+ English words): "
        f"{len(english_candidates)}"
    )

    print("\nCreated files:")
    print(f"- {ALL_RECORDS_CSV.resolve()}")
    print(f"- {ENGLISH_CANDIDATES_CSV.resolve()}")
    print(f"- {PROFILE_JSON.resolve()}")


if __name__ == "__main__":
    main()