import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("EUROPEANA_API_KEY")
SEARCH_URL = "https://api.europeana.eu/record/v2/search.json"
OUTPUT_FILE = Path("data/raw/europeana/test/ww1_test_search.json")


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "EUROPEANA_API_KEY is missing. Add it to your .env file first."
        )

    params = {
    "wskey": API_KEY,
    "query": 'DATA_PROVIDER:"Europeana 1914-1918"',
    "reusability": "open",
    "rows": PAGE_SIZE,
    "cursor": cursor,
    "sort": "random_20260628+asc",
}

    response = requests.get(SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if data.get("success") is False:
        raise RuntimeError(f"Europeana API error: {data.get('error')}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Total matching WWI records: {data.get('totalResults')}")
    print(f"Saved raw response to: {OUTPUT_FILE}\n")


    for item in data.get("items", []):
        print("ID:", item.get("id"))
        print("Title:", item.get("title"))
        print("Provider:", item.get("dataProvider"))
        print("Type:", item.get("type"))
        print("Language:", item.get("language"))
        print("Description:", item.get("dcDescription") or item.get("description"))
        print("-" * 70)

if __name__ == "__main__":
    main()