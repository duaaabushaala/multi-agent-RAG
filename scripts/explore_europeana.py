import csv
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("EUROPEANA_API_KEY")
SEARCH_URL = "https://api.europeana.eu/record/v2/search.json"

PROVIDER_NAME = "Europeana 1914-1918"
TARGET_RECORDS = 300
PAGE_SIZE = 100

OUTPUT_DIR = Path("data/raw/europeana/exploration_sample")
RAW_OUTPUT = OUTPUT_DIR / "europeana_1914_1918_search_results.json"
CSV_OUTPUT = OUTPUT_DIR / "europeana_1914_1918_summary.csv"


def to_text(value: Any) -> str:
    """Convert Europeana values into readable text for a CSV."""
    if value is None:
        return ""

    if isinstance(value, list):
        return " | ".join(to_text(item) for item in value)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def first_non_empty(*values: Any) -> str:
    """Return the first usable field value."""
    for value in values:
        text = to_text(value).strip()
        if text:
            return text
    return ""


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "EUROPEANA_API_KEY is missing."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Project folder:", Path.cwd())
    print("Saving files to:", OUTPUT_DIR.resolve())
    print(f"Collecting up to {TARGET_RECORDS} records from {PROVIDER_NAME}...")

    all_items: list[dict[str, Any]] = []

    # For 300 records, simple pagination is enough:
    # 1–100, then 101–200, then 201–300.
    for start in range(1, TARGET_RECORDS + 1, PAGE_SIZE):
        params = {
            "wskey": API_KEY,
            "query": f'DATA_PROVIDER:"{PROVIDER_NAME}"',
            "rows": PAGE_SIZE,
            "start": start,
            "profile": "standard",
        }

        response = requests.get(SEARCH_URL, params=params, timeout=30)

        if response.status_code != 200:
            print("\nEuropeana returned an error.")
            print("Status code:", response.status_code)
            print("Response body:", response.text)
            response.raise_for_status()

        payload = response.json()

        if payload.get("success") is False:
            raise RuntimeError(
                f"Europeana API error: {payload.get('error', payload)}"
            )

        items = payload.get("items", [])

        if not items:
            print("No more records returned.")
            break

        all_items.extend(items)

        print(
            f"Retrieved {len(items)} records from start={start}. "
            f"Total collected: {len(all_items)}"
        )

        time.sleep(0.3)

    all_items = all_items[:TARGET_RECORDS]

    # Save untouched search API results.
    RAW_OUTPUT.write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, str]] = []

    for item in all_items:
        description = first_non_empty(
            item.get("dcDescription"),
            item.get("description"),
        )

        rows.append(
            {
                "record_id": to_text(item.get("id")),
                "title": to_text(item.get("title")),
                "data_provider": to_text(item.get("dataProvider")),
                "provider": to_text(item.get("provider")),
                "collection_name": to_text(item.get("collectionName")),
                "record_type": to_text(item.get("type")),
                "language": to_text(item.get("language")),
                "rights": to_text(item.get("rights")),
                "description_preview": description[:500],
                "europeana_url": (
                    "https://www.europeana.eu/en/item"
                    + to_text(item.get("id"))
                ),
            }
        )

    if not rows:
        raise RuntimeError(
            "The API returned zero records. Check the provider name or query."
        )

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    type_counts = Counter(
        row["record_type"] or "Unknown"
        for row in rows
    )

    language_counts = Counter(
        row["language"] or "Unknown"
        for row in rows
    )

    descriptions_present = sum(
        1 for row in rows
        if row["description_preview"].strip()
    )

    print("\n--- Success ---")
    print(f"Records saved: {len(rows)}")
    print(f"Raw JSON: {RAW_OUTPUT.resolve()}")
    print(f"Summary CSV: {CSV_OUTPUT.resolve()}")
    print(
        f"Description coverage: {descriptions_present}/{len(rows)} "
        f"({descriptions_present / len(rows) * 100:.1f}%)"
    )

    print("\nRecord types:")
    for label, count in type_counts.most_common(10):
        print(f"{count:>4}  {label}")

    print("\nLanguage values:")
    for label, count in language_counts.most_common(10):
        print(f"{count:>4}  {label}")


if __name__ == "__main__":
    main()