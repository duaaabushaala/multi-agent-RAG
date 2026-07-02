"""
Create the final structured Europeana corpus and a shared chunk set.

Inputs:
- 230 eligible English-description candidate records
- the validated list of 219 final record IDs
- candidate document-type routing tags

Outputs:
- final_corpus_records.jsonl: one clean structured record per final item
- shared_chunks_v1.jsonl: one shared description-chunk set for both systems
- preprocessing_manifest_v1.json: counts, settings and provenance details

The raw Europeana API response is never read here.
"""

import csv
import json
import re
from pathlib import Path
from statistics import median


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

CANDIDATE_CSV = Path(
    "data/cleaned/ww1_exploration_english_candidates.csv"
)

VALIDATED_SELECTION_JSONL = Path(
    "data/processed/agent_corpora_validated/"
    "single_agent_documents.jsonl"
)

DOCUMENT_TYPE_TAGS_CSV = Path(
    "data/validation/document_type_tags_v2/"
    "record_document_type_tags.csv"
)

OUTPUT_DIR = Path("data/processed/shared_corpus")

FINAL_RECORDS_OUTPUT = OUTPUT_DIR / "final_corpus_records.jsonl"
SHARED_CHUNKS_OUTPUT = OUTPUT_DIR / "shared_chunks_v1.jsonl"
MANIFEST_OUTPUT = OUTPUT_DIR / "preprocessing_manifest_v1.json"


# -------------------------------------------------------------------
# Chunking settings: v1 development configuration
# -------------------------------------------------------------------

TARGET_CHUNK_WORDS = 220
OVERLAP_WORDS = 40


# -------------------------------------------------------------------
# Text helpers
# -------------------------------------------------------------------

def clean_text(value: object | None) -> str:
    """
    Standardise whitespace without rewriting the source wording.
    """
    if value is None:
        return ""

    text = str(value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    """Return an approximate word count."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def split_pipe_values(value: object | None) -> list[str]:
    """
    Convert Europeana's pipe-separated fields into clean lists. e.g. 1914 | 1918 become ["1914", "1918"].
    """
    if value is None:
        return []

    return [
        clean_text(part)
        for part in str(value).split("|")
        if clean_text(part)
    ]


def parse_bool(value: object | None) -> bool:
    """Read boolean-like CSV values safely."""
    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def split_into_units(text: str) -> list[str]:
    """
    Split a description into approximately sentence-sized units.
    """
    text = clean_text(text)

    if not text:
        return []

    units = re.split(
        r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9\"“‘'])",
        text,
    )

    return [
        clean_text(unit)
        for unit in units
        if clean_text(unit)
    ]


def split_long_unit(unit: str) -> list[str]:
    """
    Split unusually long sentences by word count.

    Each section is capped below the full target size so a later
    overlap does not create oversized chunks.
    """
    words = unit.split()
    max_new_words = TARGET_CHUNK_WORDS - OVERLAP_WORDS

    if len(words) <= max_new_words:
        return [unit]

    sections = []

    for start in range(0, len(words), max_new_words):
        sections.append(" ".join(words[start:start + max_new_words]))

    return sections


def create_chunks(description: str) -> list[str]:
    """
    this is the core chunking logic 
    
    Chunks are created only from English descriptive text.
    Catalogue metadata is added separately as a compact header.
    """
    units = []

    for unit in split_into_units(description):
        units.extend(split_long_unit(unit))

    chunks = []
    current_words: list[str] = []

    for unit in units:
        unit_words = unit.split()

        if (
            current_words
            and len(current_words) + len(unit_words)
            > TARGET_CHUNK_WORDS
        ):
            chunks.append(" ".join(current_words))

            overlap = current_words[-OVERLAP_WORDS:]
            current_words = overlap.copy()

        current_words.extend(unit_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [
        clean_text(chunk)
        for chunk in chunks
        if clean_text(chunk)
    ]


# -------------------------------------------------------------------
# Input loading
# -------------------------------------------------------------------

def load_validated_record_ids() -> list[str]:
    """
    Load the exact 219 record IDs retained after validation.
    """
    if not VALIDATED_SELECTION_JSONL.exists():
        raise FileNotFoundError(
            f"Missing validated selection file: "
            f"{VALIDATED_SELECTION_JSONL}"
        )

    record_ids = []

    with VALIDATED_SELECTION_JSONL.open(
        encoding="utf-8"
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                record = json.loads(line)
                record_id = clean_text(record.get("record_id"))

                if not record_id:
                    raise ValueError(
                        "A validated selection row has no record_id."
                    )

                record_ids.append(record_id)

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "Duplicate record IDs found in validated selection file."
        )

    if len(record_ids) != 219:
        raise ValueError(
            "Expected 219 validated records, but found "
            f"{len(record_ids)}."
        )

    return record_ids


def load_candidate_rows() -> dict[str, dict]:
    """
    Load candidate records into a lookup keyed by record ID.
    """
    if not CANDIDATE_CSV.exists():
        raise FileNotFoundError(
            f"Missing candidate CSV: {CANDIDATE_CSV}"
        )

    rows_by_id = {}

    with CANDIDATE_CSV.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            record_id = clean_text(row.get("record_id"))

            if not record_id:
                continue

            if record_id in rows_by_id:
                raise ValueError(
                    f"Duplicate candidate record ID: {record_id}"
                )

            rows_by_id[record_id] = row

    if len(rows_by_id) != 230:
        print(
            "Warning: expected 230 candidate records, but found "
            f"{len(rows_by_id)}."
        )

    return rows_by_id


def load_document_type_tags() -> dict[str, dict]:
    """
    Load candidate routing tags for the 219-record corpus.

    These are routing metadata, not historical ground truth.
    """
    if not DOCUMENT_TYPE_TAGS_CSV.exists():
        raise FileNotFoundError(
            f"Missing document-type tag CSV: "
            f"{DOCUMENT_TYPE_TAGS_CSV}"
        )

    tags_by_id = {}

    with DOCUMENT_TYPE_TAGS_CSV.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            record_id = clean_text(row.get("record_id"))

            if not record_id:
                continue

            if record_id in tags_by_id:
                raise ValueError(
                    f"Duplicate tag record ID: {record_id}"
                )

            tag_labels = []

            for field in [
                "personal_history",
                "written_materials",
                "visual_object_materials",
            ]:
                if parse_bool(row.get(field)):
                    tag_labels.append(field)

            tags_by_id[record_id] = {
                "labels": tag_labels,
                "combination": clean_text(
                    row.get("tag_combination")
                ),
                "evidence": {
                    "personal_history": clean_text(
                        row.get("personal_history_evidence")
                    ),
                    "written_materials": clean_text(
                        row.get("written_materials_evidence")
                    ),
                    "visual_object_materials": clean_text(
                        row.get(
                            "visual_object_materials_evidence"
                        )
                    ),
                },
            }

    return tags_by_id


# -------------------------------------------------------------------
# Record and chunk construction
# -------------------------------------------------------------------

def make_catalogue_metadata(row: dict) -> dict:
    """Create a structured catalogue metadata object."""
    return {
        "title": clean_text(row.get("title_variants")),
        "record_type": clean_text(row.get("record_type")),
        "years": split_pipe_values(row.get("year_values")),
        "places": split_pipe_values(row.get("place_values")),
        "subjects": split_pipe_values(row.get("english_subjects")),
        "original_item_language": split_pipe_values(
            row.get("original_item_language")
        ),
        "provider": clean_text(row.get("provider")),
        "data_provider": clean_text(row.get("data_provider")),
        "collection_name": clean_text(
            row.get("collection_name")
        ),
        "rights": clean_text(row.get("rights")),
    }


def make_catalogue_header(metadata: dict) -> str:
    """
    Make the compact header added to each description chunk.

    Short fields remain attached to meaningful description evidence,
    rather than becoming standalone tiny chunks.
    """
    lines = []

    title = metadata.get("title", "")
    if title:
        lines.append(f"Title: {title}")

    record_type = metadata.get("record_type", "")
    if record_type:
        lines.append(f"Record type: {record_type}")

    years = metadata.get("years", [])
    if years:
        lines.append(f"Years: {'; '.join(years)}")

    places = metadata.get("places", [])
    if places:
        lines.append(f"Places: {'; '.join(places)}")

    subjects = metadata.get("subjects", [])
    if subjects:
        lines.append(f"Subjects: {'; '.join(subjects)}")

    language = metadata.get("original_item_language", [])
    if language:
        lines.append(
            f"Original item language: {'; '.join(language)}"
        )

    provider = metadata.get("provider", "")
    if provider:
        lines.append(f"Provider: {provider}")

    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL with UTF-8 text preserved."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=False,
                )
                + "\n"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading validated selection...")
    validated_ids = load_validated_record_ids()

    print("Loading candidate records...")
    candidates_by_id = load_candidate_rows()

    print("Loading document-type routing tags...")
    tags_by_id = load_document_type_tags()

    missing_candidates = [
        record_id
        for record_id in validated_ids
        if record_id not in candidates_by_id
    ]

    missing_tags = [
        record_id
        for record_id in validated_ids
        if record_id not in tags_by_id
    ]

    if missing_candidates:
        raise ValueError(
            "Validated records missing from candidate CSV:\n"
            + "\n".join(missing_candidates)
        )

    if missing_tags:
        raise ValueError(
            "Validated records missing from tag CSV:\n"
            + "\n".join(missing_tags)
        )

    processed_records = []
    chunk_rows = []
    chunks_per_record = []
    description_word_counts = []
    tag_counts = {
        "personal_history": 0,
        "written_materials": 0,
        "visual_object_materials": 0,
    }

    for record_id in validated_ids:
        row = candidates_by_id[record_id]
        routing_tags = tags_by_id[record_id]

        catalogue_metadata = make_catalogue_metadata(row)
        title = catalogue_metadata["title"]
        descriptive_text = clean_text(row.get("english_text_all"))
        provenance_url = clean_text(row.get("europeana_url"))

        if not title:
            raise ValueError(
                f"Missing title for final record: {record_id}"
            )

        if not descriptive_text:
            raise ValueError(
                f"Missing English descriptive text: {record_id}"
            )

        if not provenance_url:
            raise ValueError(
                f"Missing Europeana provenance URL: {record_id}"
            )

        for label in routing_tags["labels"]:
            tag_counts[label] += 1

        description_word_counts.append(
            word_count(descriptive_text)
        )

        processed_record = {
            "schema_version": "1.0",
            "document_id": record_id,
            "record_id": record_id,
            "title": title,
            "source": "Europeana 1914-1918 catalogue record",
            "provenance_url": provenance_url,
            "field_groups": {
                "catalogue_metadata": catalogue_metadata,
                "english_descriptive_text": descriptive_text,
                "external_context": None,
            },
            "external_context_used": False,
            "description_source_field": "english_text_all",
            "document_type_tags": {
                "labels": routing_tags["labels"],
                "candidate_routing_metadata": True,
                "combination": routing_tags["combination"],
                "evidence": routing_tags["evidence"],
            },
        }

        processed_records.append(processed_record)

        catalogue_header = make_catalogue_header(
            catalogue_metadata
        )
        chunks = create_chunks(descriptive_text)

        if not chunks:
            raise ValueError(
                f"No chunks created for record: {record_id}"
            )

        chunks_per_record.append(len(chunks))

        for chunk_index, chunk_text in enumerate(
            chunks,
            start=1,
        ):
            chunk_id = (
                f"{record_id}::description::"
                f"chunk_{chunk_index:03d}"
            )

            retrieval_text = (
                f"{catalogue_header}\n\n"
                f"Description:\n{chunk_text}"
            )

            chunk_rows.append(
                {
                    "schema_version": "1.0",
                    "chunk_id": chunk_id,
                    "document_id": record_id,
                    "record_id": record_id,
                    "chunk_index": chunk_index,
                    "field_group": "english_descriptive_text",
                    "catalogue_metadata_attached": True,
                    "source": (
                        "Europeana 1914-1918 catalogue record"
                    ),
                    "provenance_url": provenance_url,
                    "title": title,
                    "catalogue_metadata": catalogue_metadata,
                    "document_type_tags": routing_tags["labels"],
                    "document_type_tag_evidence": (
                        routing_tags["evidence"]
                    ),
                    "chunk_text": chunk_text,
                    "chunk_word_count": word_count(chunk_text),
                    "embedding_text": retrieval_text,
                    "retrieval_text": retrieval_text,
                }
            )

    write_jsonl(FINAL_RECORDS_OUTPUT, processed_records)
    write_jsonl(SHARED_CHUNKS_OUTPUT, chunk_rows)

    manifest = {
        "schema_version": "1.0",
        "selection": {
            "candidate_records_available": len(
                candidates_by_id
            ),
            "final_validated_records": len(processed_records),
            "candidate_records_not_selected": (
                len(candidates_by_id) - len(processed_records)
            ),
            "selection_source": str(
                VALIDATED_SELECTION_JSONL
            ),
        },
        "source_files": {
            "candidate_records_csv": str(CANDIDATE_CSV),
            "validated_selection_jsonl": str(
                VALIDATED_SELECTION_JSONL
            ),
            "document_type_tags_csv": str(
                DOCUMENT_TYPE_TAGS_CSV
            ),
        },
        "preprocessing": {
            "text_cleaning": (
                "Whitespace was standardised while preserving "
                "the original English catalogue wording."
            ),
            "field_groups": {
                "catalogue_metadata": (
                    "Structured Europeana catalogue fields."
                ),
                "english_descriptive_text": (
                    "Primary retrieval evidence."
                ),
                "external_context": (
                    "Not used in this experiment."
                ),
            },
            "short_field_approach": (
                "Short catalogue fields are not standalone chunks. "
                "They are stored as metadata and attached as a "
                "compact header to each description chunk."
            ),
            "document_type_tags": (
                "Candidate routing metadata only; not historical "
                "ground truth."
            ),
            "fairness_rule": (
                "This one shared chunk set is the common evidence "
                "base for the single-agent and multi-agent systems."
            ),
        },
        "chunking_v1": {
            "chunk_source_field": "english_text_all",
            "method": (
                "Sentence-aware approximate word chunking with "
                "word-based overlap."
            ),
            "target_chunk_words": TARGET_CHUNK_WORDS,
            "overlap_words": OVERLAP_WORDS,
            "total_chunks": len(chunk_rows),
            "chunks_per_record": {
                "minimum": min(chunks_per_record),
                "median": median(chunks_per_record),
                "maximum": max(chunks_per_record),
            },
            "chunk_word_counts": {
                "minimum": min(
                    row["chunk_word_count"]
                    for row in chunk_rows
                ),
                "median": median(
                    row["chunk_word_count"]
                    for row in chunk_rows
                ),
                "maximum": max(
                    row["chunk_word_count"]
                    for row in chunk_rows
                ),
            },
            "description_word_counts": {
                "minimum": min(description_word_counts),
                "median": median(description_word_counts),
                "maximum": max(description_word_counts),
            },
        },
        "candidate_document_type_tag_counts": tag_counts,
        "outputs": {
            "processed_records": str(FINAL_RECORDS_OUTPUT),
            "shared_chunks_v1": str(SHARED_CHUNKS_OUTPUT),
        },
    }

    MANIFEST_OUTPUT.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n--- Shared corpus v1 created ---")
    print(f"Final records: {len(processed_records)}")
    print(f"Shared chunks: {len(chunk_rows)}")
    print(
        "Chunks per record: "
        f"min={min(chunks_per_record)}, "
        f"median={median(chunks_per_record)}, "
        f"max={max(chunks_per_record)}"
    )
    print(
        "Chunk words: "
        f"min={manifest['chunking_v1']['chunk_word_counts']['minimum']}, "
        f"median={manifest['chunking_v1']['chunk_word_counts']['median']}, "
        f"max={manifest['chunking_v1']['chunk_word_counts']['maximum']}"
    )
    print("\nCreated:")
    print(f"- {FINAL_RECORDS_OUTPUT}")
    print(f"- {SHARED_CHUNKS_OUTPUT}")
    print(f"- {MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()