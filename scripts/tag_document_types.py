# scripts/tag_document_types.py

"""
Create revised candidate document-type tags for the validated Europeana corpus.

This script:
- keeps records whole;
- allows overlapping document-type tags;
- uses only the actual Europeana descriptive text for personal-history tags;
- uses titles plus descriptive text for written/visual-object material tags;
- writes into a new v2 folder, preserving the earlier audit outputs.

Tags:
- personal_history
- written_materials
- visual_object_materials
"""

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SOURCE_JSONL = Path(
    "data/processed/agent_corpora_validated/"
    "single_agent_documents.jsonl"
)

# New folder: does not overwrite the earlier tagging output.
OUTPUT_DIR = Path("data/validation/document_type_tags_v2")

TAGS_OUTPUT = OUTPUT_DIR / "record_document_type_tags.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "tag_summary.json"
SAMPLE_OUTPUT = OUTPUT_DIR / "tag_audit_sample.csv"

RANDOM_SEED = 20260629
AUDIT_SAMPLE_SIZE = 15


# --------------------------------------------------
# Tagging patterns
# --------------------------------------------------

# Personal history should be based on direct biographical, service,
# injury, captivity, death, or post-war evidence about an individual.
# Generic mentions of soldiers or a family member are not enough alone.

PERSONAL_HISTORY_PATTERNS = {
    "biographical_event": (
        r"\b(?:born|died|killed|missing in action|"
        r"survived|returned from the war|married|"
        r"demobilized|demobilised)\b"
    ),
    "service_event": (
        r"\b(?:served|fought|enlisted|mobilized|mobilised|"
        r"drafted|called up|joined|assigned|stationed|"
        r"volunteered|voluntarily offered|"
        r"taken prisoner|captured)\b"
    ),
    "war_harm_or_outcome": (
        r"\b(?:wounded|injured|gassed|hospitalized|"
        r"hospitalised|evacuated|decorated|cited|"
        r"received the cross|received the medal)\b"
    ),
    "military_role_or_unit": (
        r"\b(?:private|corporal|sergeant|lieutenant|captain|"
        r"major|colonel|commander|gunner|driver|"
        r"stretcher[- ]bearer|nurse[- ]sergeant)\b|"
        r"\b(?:regiment|battalion|company|division|"
        r"army corps|artillery|infantry)\b"
    ),
    "war_setting": (
        r"\b(?:front|trench|trenches|battle|campaign|"
        r"captivity|prison camp)\b"
    ),
}

WRITTEN_MATERIAL_PATTERNS = {
    "letters_or_correspondence": (
        r"\b(?:letter|letters|correspondence|"
        r"written to|wrote to)\b"
    ),
    "diary_or_journal": (
        r"\b(?:diary|diaries|journal|journals|"
        r"notebook|notebooks|memoir|memoirs)\b"
    ),
    "manuscript_or_transcription": (
        r"\b(?:manuscript|manuscripts|transcript|"
        r"transcription|written account|written story|"
        r"notes)\b"
    ),
    "postcards_or_telegram": (
        r"\b(?:postcard|postcards|telegram|telegrams)\b"
    ),
}

VISUAL_OBJECT_PATTERNS = {
    "photographic_material": (
        r"\b(?:photograph|photographs|photo|photos|"
        r"photographic|album|albums|glass plates?)\b"
    ),
    "postcard_or_map": (
        r"\b(?:postcard|postcards|map|maps)\b"
    ),
    "medal_or_uniform": (
        r"\b(?:medal|medals|decoration|decorations|"
        r"uniform|uniforms|helmet|helmets|badge|badges|"
        r"certificate|certificates)\b"
    ),
    "object_or_artwork": (
        r"\b(?:object|objects|artefact|artefacts|artifact|"
        r"artifacts|painting|paintings|drawing|drawings|"
        r"sketch|sketches|portrait|portraits|flag|flags|"
        r"booklet|booklets)\b"
    ),
}


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def clean_text(value: Any) -> str:
    """Return clean, single-spaced text."""
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def word_count(text: str) -> int:
    """Return an approximate word count."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def extract_descriptive_text(full_text: str) -> str:
    """
    Extract only the Europeana descriptive-text section.

    Baseline documents were built in this form:

    TITLE
    ...
    STRUCTURED METADATA
    ...
    DESCRIPTIVE TEXT
    actual Europeana description
    """
    match = re.search(
        r"(?is)\bDESCRIPTIVE\s+TEXT\s*(.*)$",
        full_text,
    )

    if match:
        return clean_text(match.group(1))

    # Fallback: retain all text if an older document format differs.
    return clean_text(full_text)


def matched_categories(
    text: str,
    patterns: dict[str, str],
) -> list[str]:
    """Return labels for each pattern group found in text."""
    matches: list[str] = []

    for label, pattern in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(label)

    return matches


def tag_combination(tags: dict[str, bool]) -> str:
    """Create a readable tag-combination label."""
    selected = [
        tag_name
        for tag_name, enabled in tags.items()
        if enabled
    ]

    return " + ".join(selected) if selected else "no_tags"


# --------------------------------------------------
# Tagging logic
# --------------------------------------------------

def tag_record(
    title: str,
    descriptive_text: str,
) -> dict[str, Any]:
    """
    Produce overlapping candidate tags plus transparent reasons.

    Personal-history evidence comes only from descriptive text.
    Written and visual/object material tags may also use titles,
    because catalogue titles can accurately name a diary, album,
    letter collection, medal, photograph, etc.
    """
    title_and_description = (
        f"{title}\n{descriptive_text}"
    )

    personal_evidence = matched_categories(
        descriptive_text,
        PERSONAL_HISTORY_PATTERNS,
    )

    written_evidence = matched_categories(
        title_and_description,
        WRITTEN_MATERIAL_PATTERNS,
    )

    visual_object_evidence = matched_categories(
        title_and_description,
        VISUAL_OBJECT_PATTERNS,
    )

    direct_personal_categories = {
        "biographical_event",
        "service_event",
        "war_harm_or_outcome",
    }

    # A record becomes personal-history only if it contains at least
    # one direct life/service/war-outcome signal. Generic terms such
    # as "soldier", "family", "battle", or "front" cannot tag it alone.
    personal_history = any(
        category in direct_personal_categories
        for category in personal_evidence
    )

    written_materials = len(written_evidence) > 0
    visual_object_materials = len(visual_object_evidence) > 0

    return {
        "personal_history": personal_history,
        "written_materials": written_materials,
        "visual_object_materials": visual_object_materials,
        "personal_history_evidence": "; ".join(
            personal_evidence
        ),
        "written_materials_evidence": "; ".join(
            written_evidence
        ),
        "visual_object_materials_evidence": "; ".join(
            visual_object_evidence
        ),
    }


# --------------------------------------------------
# File handling
# --------------------------------------------------

def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write a CSV file with a header."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_baseline_documents() -> list[dict[str, Any]]:
    """Load the validated baseline documents."""
    if not SOURCE_JSONL.exists():
        raise FileNotFoundError(
            "Could not find the validated baseline file:\n"
            f"{SOURCE_JSONL.resolve()}\n\n"
            "Run prepare_agent_corpora.py first."
        )

    documents: list[dict[str, Any]] = []

    with SOURCE_JSONL.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                documents.append(json.loads(line))

    if not documents:
        raise RuntimeError(
            "The validated baseline JSONL contains no documents."
        )

    return documents


def make_audit_sample(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Create a reproducible sample across each tag combination.

    Up to three records are selected per combination, then remaining
    slots are filled randomly until the target sample size is reached.
    """
    rng = random.Random(RANDOM_SEED)

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped_rows[row["tag_combination"]].append(row)

    selected_rows: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    for combination in sorted(grouped_rows):
        group = grouped_rows[combination]
        amount = min(3, len(group))

        for row in rng.sample(group, amount):
            selected_rows.append(row)
            selected_ids.add(row["record_id"])

    remaining_rows = [
        row
        for row in rows
        if row["record_id"] not in selected_ids
    ]

    remaining_slots = max(
        AUDIT_SAMPLE_SIZE - len(selected_rows),
        0,
    )

    if remaining_slots > 0 and remaining_rows:
        selected_rows.extend(
            rng.sample(
                remaining_rows,
                min(remaining_slots, len(remaining_rows)),
            )
        )

    audit_rows: list[dict[str, str]] = []

    for sample_number, row in enumerate(selected_rows, start=1):
        audit_rows.append(
            {
                "sample_id": f"T{sample_number:02d}",
                "record_id": row["record_id"],
                "title": row["title"],
                "tag_combination": row["tag_combination"],
                "personal_history": row["personal_history"],
                "written_materials": row["written_materials"],
                "visual_object_materials": (
                    row["visual_object_materials"]
                ),
                "personal_history_evidence": (
                    row["personal_history_evidence"]
                ),
                "written_materials_evidence": (
                    row["written_materials_evidence"]
                ),
                "visual_object_materials_evidence": (
                    row["visual_object_materials_evidence"]
                ),
                "descriptive_text": row["descriptive_text"],
                "review_personal_history_correct_yes_no": "",
                "review_written_materials_correct_yes_no": "",
                "review_visual_object_correct_yes_no": "",
                "review_notes": "",
            }
        )

    return audit_rows


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading validated baseline documents...")
    documents = load_baseline_documents()

    tag_rows: list[dict[str, str]] = []

    for document in documents:
        record_id = clean_text(document.get("record_id"))
        title = clean_text(document.get("title"))

        # Important: do not clean the full text until AFTER extracting
        # the descriptive-text section, otherwise line structure is lost.
        full_document_text = str(document.get("text", ""))

        descriptive_text = extract_descriptive_text(
            full_document_text
        )

        tag_result = tag_record(title, descriptive_text)

        tags = {
            "personal_history": bool(
                tag_result["personal_history"]
            ),
            "written_materials": bool(
                tag_result["written_materials"]
            ),
            "visual_object_materials": bool(
                tag_result["visual_object_materials"]
            ),
        }

        tag_rows.append(
            {
                "record_id": record_id,
                "title": title,
                "descriptive_word_count": str(
                    word_count(descriptive_text)
                ),
                "personal_history": str(
                    tags["personal_history"]
                ),
                "written_materials": str(
                    tags["written_materials"]
                ),
                "visual_object_materials": str(
                    tags["visual_object_materials"]
                ),
                "tag_combination": tag_combination(tags),
                "personal_history_evidence": str(
                    tag_result["personal_history_evidence"]
                ),
                "written_materials_evidence": str(
                    tag_result["written_materials_evidence"]
                ),
                "visual_object_materials_evidence": str(
                    tag_result[
                        "visual_object_materials_evidence"
                    ]
                ),
                "descriptive_text": descriptive_text,
            }
        )

    tag_counts = {
        "personal_history": sum(
            row["personal_history"] == "True"
            for row in tag_rows
        ),
        "written_materials": sum(
            row["written_materials"] == "True"
            for row in tag_rows
        ),
        "visual_object_materials": sum(
            row["visual_object_materials"] == "True"
            for row in tag_rows
        ),
    }

    combination_counts = Counter(
        row["tag_combination"]
        for row in tag_rows
    )

    no_tag_records = [
        row["record_id"]
        for row in tag_rows
        if row["tag_combination"] == "no_tags"
    ]

    audit_sample = make_audit_sample(tag_rows)

    write_csv(
        TAGS_OUTPUT,
        tag_rows,
        fieldnames=[
            "record_id",
            "title",
            "descriptive_word_count",
            "personal_history",
            "written_materials",
            "visual_object_materials",
            "tag_combination",
            "personal_history_evidence",
            "written_materials_evidence",
            "visual_object_materials_evidence",
            "descriptive_text",
        ],
    )

    write_csv(
        SAMPLE_OUTPUT,
        audit_sample,
        fieldnames=[
            "sample_id",
            "record_id",
            "title",
            "tag_combination",
            "personal_history",
            "written_materials",
            "visual_object_materials",
            "personal_history_evidence",
            "written_materials_evidence",
            "visual_object_materials_evidence",
            "descriptive_text",
            "review_personal_history_correct_yes_no",
            "review_written_materials_correct_yes_no",
            "review_visual_object_correct_yes_no",
            "review_notes",
        ],
    )

    summary = {
        "corpus_records_tagged": len(tag_rows),
        "tagging_status": (
            "revised_candidate_tags_pending_manual_audit"
        ),
        "text_extraction_method": (
            "Only text after the DESCRIPTIVE TEXT marker was used "
            "for personal-history tagging."
        ),
        "personal_history_rule": (
            "At least one direct biographical, service, or war-outcome "
            "signal was required. Generic military/family references "
            "could not create this tag alone."
        ),
        "tag_counts": tag_counts,
        "tag_combination_counts": dict(
            sorted(combination_counts.items())
        ),
        "records_with_no_candidate_tags": len(no_tag_records),
        "record_ids_with_no_candidate_tags": no_tag_records,
        "audit_sample_size": len(audit_sample),
        "audit_sampling_method": (
            "Up to three random records per tag combination, "
            "then random records to fill the target sample size."
        ),
        "random_seed": RANDOM_SEED,
        "outputs": {
            "all_candidate_tags": str(TAGS_OUTPUT),
            "manual_audit_sample": str(SAMPLE_OUTPUT),
        },
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n--- Revised candidate document-type tags created ---")
    print(f"Records tagged: {len(tag_rows)}")
    print(
        "Personal-history records: "
        f"{tag_counts['personal_history']}"
    )
    print(
        "Written-material records: "
        f"{tag_counts['written_materials']}"
    )
    print(
        "Visual/object-material records: "
        f"{tag_counts['visual_object_materials']}"
    )
    print(
        "Records with no candidate tags: "
        f"{len(no_tag_records)}"
    )
    print(f"Audit sample created: {len(audit_sample)} records")

    print("\nCreated:")
    print(f"- {TAGS_OUTPUT.resolve()}")
    print(f"- {SAMPLE_OUTPUT.resolve()}")
    print(f"- {SUMMARY_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()