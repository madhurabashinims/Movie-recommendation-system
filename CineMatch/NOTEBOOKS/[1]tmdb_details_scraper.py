#!/usr/bin/env python3
"""
tmdb_details_scraper_phase2.py

Phase 2 scraper: read movie_tmdb_mapping.csv and fetch TMDb movie details for matched tmdb_id.
Produces movie_tmdb_details.csv and saves raw JSON per tmdb_id.

Features:
- Batch processing
- Resume capability: skips tmdb_ids already present in output CSV
- Robust retries with exponential backoff + jitter
- Checkpoint saving after every batch
- Optional saving of raw JSON responses for debugging

Requirements:
- Python 3.8+
- pandas
- requests
- (optional) tqdm for progress bar

Run:
    python tmdb_details_scraper_phase2.py

Edit CONFIG below to set API_KEY, batch size, file names, etc.
"""

import os
import time
import json
import csv
import math
import random
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import requests
import pandas as pd

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# ============================
# CONFIG - change these
# ============================
API_KEY = "656efa157ee0f47cae123095d9c0564e"   # <--- Put your TMDb API key here or set env var
MAPPING_CSV = "D:/MINI PROJECT/NOTEBOOKS/movie_tmdb_mapping.csv"        # input mapping (must exist)
OUTPUT_CSV = "movie_tmdb_details.csv"         # output details CSV (resume-able)
RAW_JSON_DIR = Path("raw_tmdb_json")          # optional: save raw JSON here for each tmdb_id
RAW_JSON_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 500            # how many movies processed per checkpoint save
SLEEP_BETWEEN_CALLS = 0.6   # seconds between API calls (be kind to TMDb)
TIMEOUT = 15                # seconds per HTTP request
MAX_RETRIES = 4             # number of retries per movie (with backoff)
BATCH_LIMIT = None          # set to an integer to limit number of remaining movies processed in this run
FORCE_FETCH = False         # if True, re-fetch even if tmdb_id exists in output (useful for debug)
VERBOSE = True
# ============================

# If env variable is set, prefer it (secure)
if os.getenv("TMDB_API_KEY"):
    API_KEY = os.getenv("TMDB_API_KEY")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TMDB-Scraper/1.0; +https://example.com)"
}

# Force IPv4 on some networks (optional)
try:
    import requests.packages.urllib3.util.connection as urllib3_cn
    urllib3_cn.HAS_IPV6 = False
except Exception:
    pass

# ============================
# Helper functions
# ============================

def safe_request_json(url: str, timeout: int = TIMEOUT, max_retries: int = MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """
    GET url and return JSON, with retries and exponential backoff + jitter.
    Returns None if all attempts fail.
    """
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if VERBOSE:
                print(f"  [request] attempt {attempt} failed: {repr(e)}")
            if attempt < max_retries:
                sleep_time = backoff + random.uniform(0, 0.5)
                if VERBOSE:
                    print(f"    sleeping {sleep_time:.2f}s before retry")
                time.sleep(sleep_time)
                backoff *= 2
            else:
                if VERBOSE:
                    print("    all retries failed for this request.")
                return None
    return None


def extract_movie_fields(movieId: str, tmdb_id: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten TMDb JSON 'details' into a tidy dict suitable for CSV.
    If details is None, produce a row with status='fetch_failed'
    """
    if details is None:
        return {
            "movieId": movieId,
            "tmdb_id": tmdb_id,
            "title": None,
            "original_title": None,
            "overview": None,
            "tagline": None,
            "genres": None,
            "keywords": None,
            "director": None,
            "cast_top5": None,
            "runtime": None,
            "release_date": None,
            "release_year": None,
            "popularity": None,
            "vote_average": None,
            "vote_count": None,
            "poster_path": None,
            "status": "fetch_failed"
        }

    # Basic fields
    title = details.get("title")
    original_title = details.get("original_title")
    overview = details.get("overview")
    tagline = details.get("tagline")
    runtime = details.get("runtime")
    popularity = details.get("popularity")
    vote_average = details.get("vote_average")
    vote_count = details.get("vote_count")
    release_date = details.get("release_date")
    release_year = None
    if release_date:
        try:
            release_year = int(str(release_date)[:4])
        except Exception:
            release_year = None
    poster_path = details.get("poster_path")

    # Genres: list of dicts
    genres_list = details.get("genres") or []
    genres = ", ".join([g.get("name", "") for g in genres_list if g.get("name")])

    # Keywords may be under details['keywords'] which has 'keywords' key
    kw_block = details.get("keywords") or {}
    if isinstance(kw_block, dict):
        kw_list = kw_block.get("keywords") or kw_block.get("results") or []
    else:
        kw_list = kw_block or []
    keywords = ", ".join([k.get("name", "") for k in kw_list if k.get("name")])

    # Credits: cast and crew
    credits = details.get("credits") or {}
    cast_list = credits.get("cast") or []
    crew_list = credits.get("crew") or []

    cast_names = [c.get("name") for c in cast_list if c.get("name")]
    cast_top5 = ", ".join(cast_names[:5]) if cast_names else None

    director_names = [c.get("name") for c in crew_list if c.get("job") == "Director" and c.get("name")]
    director = ", ".join(director_names) if director_names else None

    return {
        "movieId": movieId,
        "tmdb_id": tmdb_id,
        "title": title,
        "original_title": original_title,
        "overview": overview,
        "tagline": tagline,
        "genres": genres,
        "keywords": keywords,
        "director": director,
        "cast_top5": cast_top5,
        "runtime": runtime,
        "release_date": release_date,
        "release_year": release_year,
        "popularity": popularity,
        "vote_average": vote_average,
        "vote_count": vote_count,
        "poster_path": poster_path,
        "status": "ok"
    }


# ============================
# Main pipeline
# ============================

def main():
    # 0. sanity checks
    if not API_KEY or len(API_KEY.strip()) < 10:
        print("ERROR: Please set a valid TMDB API_KEY in the script or via TMDB_API_KEY env var.")
        return

    if not os.path.exists(MAPPING_CSV):
        print(f"ERROR: mapping file {MAPPING_CSV} not found in current folder.")
        return

    # 1. read mapping CSV
    print("Loading mapping:", MAPPING_CSV)
    mapping = pd.read_csv(MAPPING_CSV, dtype=str)
    # ensure columns exist
    for col in ["movieId", "tmdb_id", "match_confidence"]:
        if col not in mapping.columns:
            print(f"ERROR: mapping CSV missing required column: {col}")
            return

    # filter accepted matches (you can change accepted_conf if you want)
    accepted_conf = {"exact_year", "near_exact", "title_match"}
    mapping = mapping[mapping["match_confidence"].isin(accepted_conf)].copy()
    mapping = mapping[mapping["tmdb_id"].notna()].copy()
    mapping["tmdb_id"] = mapping["tmdb_id"].astype(str)
    mapping["movieId"] = mapping["movieId"].astype(str)

    print("Total accepted mapping rows:", len(mapping))

    # 2. load existing details if present (to resume)
    if os.path.exists(OUTPUT_CSV) and not FORCE_FETCH:
        try:
            existing = pd.read_csv(OUTPUT_CSV, dtype=str)
            processed_tmdb = set(existing["tmdb_id"].astype(str).tolist())
            print("Existing details rows:", len(existing))
        except Exception as e:
            print("Warning: failed to read existing output CSV (will re-create):", e)
            existing = pd.DataFrame()
            processed_tmdb = set()
    else:
        existing = pd.DataFrame()
        processed_tmdb = set()
        print("No existing details file found. Creating new output.")

    # 3. build list to process
    to_process = mapping[~mapping["tmdb_id"].isin(processed_tmdb)].copy()
    print("New movies to fetch:", len(to_process))

    if to_process.empty:
        print("Nothing new to fetch. Exiting.")
        return

    # limit for this run (optional)
    if BATCH_LIMIT is not None:
        to_process = to_process.iloc[:BATCH_LIMIT].copy()
        print("Applying BATCH_LIMIT, rows to process in THIS RUN:", len(to_process))

    # 4. iterate in batches
    rows_buffer = []
    processed_count = 0
    total_to_process = len(to_process)
    iterator = to_process.iterrows()
    if TQDM_AVAILABLE:
        iterator = tqdm(to_process.iterrows(), total=total_to_process)

    base_url_template = "https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&append_to_response=credits,keywords,images"

    try:
        for idx, row in iterator:
            movieId = str(row["movieId"])
            tmdb_id = str(row["tmdb_id"])

            # safety: skip if already processed (double-check)
            if not FORCE_FETCH and tmdb_id in processed_tmdb:
                if VERBOSE:
                    print(f"[{tmdb_id}] already processed, skipping.")
                continue

            if VERBOSE:
                print(f"\nProcessing movieId={movieId} tmdb_id={tmdb_id} ...")

            url = base_url_template.format(tmdb_id=tmdb_id, api_key=API_KEY)

            details = safe_request_json(url)

            # optional: save raw JSON for debug
            if details is not None:
                try:
                    raw_path = RAW_JSON_DIR / f"{tmdb_id}.json"
                    with open(raw_path, "w", encoding="utf-8") as f:
                        json.dump(details, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    if VERBOSE:
                        print(f"  [!] failed to write raw json for {tmdb_id}: {e}")

            # extract and append
            row_out = extract_movie_fields(movieId, tmdb_id, details)
            rows_buffer.append(row_out)
            processed_tmdb.add(tmdb_id)
            processed_count += 1

            # be kind to API
            time.sleep(SLEEP_BETWEEN_CALLS)

            # checkpoint every BATCH_SIZE movies
            if processed_count % BATCH_SIZE == 0 or processed_count == total_to_process:
                # combine and save
                if not existing.empty:
                    df_existing = existing
                else:
                    df_existing = pd.DataFrame()
                df_new = pd.DataFrame(rows_buffer)
                if df_existing.empty:
                    combined = df_new
                else:
                    combined = pd.concat([df_existing, df_new], ignore_index=True)
                # drop duplicates on tmdb_id keep first
                combined = combined.drop_duplicates(subset=["tmdb_id"], keep="first")
                combined.to_csv(OUTPUT_CSV, index=False)
                if VERBOSE:
                    print(f"\nCheckpoint saved. Processed this run: {processed_count}. Total rows now: {len(combined)}")
                # update existing and clear buffer
                existing = combined
                rows_buffer = []

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Saving progress before exit...")
        # save buffer
        if rows_buffer:
            if not existing.empty:
                combined = pd.concat([existing, pd.DataFrame(rows_buffer)], ignore_index=True)
            else:
                combined = pd.DataFrame(rows_buffer)
            combined = combined.drop_duplicates(subset=["tmdb_id"], keep="first")
            combined.to_csv(OUTPUT_CSV, index=False)
            print("Progress saved to", OUTPUT_CSV)
        print("Exiting by user request.")
        return

    except Exception as e:
        print("\nUnexpected error occurred:", repr(e))
        traceback.print_exc()
        # save buffer to avoid losing data
        if rows_buffer:
            try:
                if not existing.empty:
                    combined = pd.concat([existing, pd.DataFrame(rows_buffer)], ignore_index=True)
                else:
                    combined = pd.DataFrame(rows_buffer)
                combined = combined.drop_duplicates(subset=["tmdb_id"], keep="first")
                combined.to_csv(OUTPUT_CSV, index=False)
                print("Saved progress to", OUTPUT_CSV)
            except Exception as ee:
                print("Failed to save progress:", ee)
        print("Exiting due to error. Inspect logs and retry.")
        return

    # final save if any remaining
    if rows_buffer:
        if not existing.empty:
            combined = pd.concat([existing, pd.DataFrame(rows_buffer)], ignore_index=True)
        else:
            combined = pd.DataFrame(rows_buffer)
        combined = combined.drop_duplicates(subset=["tmdb_id"], keep="first")
        combined.to_csv(OUTPUT_CSV, index=False)
        print("Final save completed. Output:", OUTPUT_CSV)

    print("\nAll done. Total newly processed in this run:", processed_count)
    print("Total rows in output file:", len(existing) if not existing.empty else processed_count)


if __name__ == "__main__":
    main()
