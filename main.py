from data.db import init_DB
from data.load_pbp import fetch_and_store

def main():
    print("--- RedZone System Initialization ---")
    
    # Phase 0: Setup Database
    print("[1/2] Initializing database...")
    init_DB()
    
    # Part 1: Data Foundation
    print("[2/2] Fetching NFL play-by-play data (2021-2024)...")
    try:
        row_count = fetch_and_store()
        print(f"Successfully loaded {row_count} plays into RedZone storage.")
    except Exception as e:
        print(f"Error loading data: {e}")

    print("\nPhase 1 Complete. Ready for Part 2: Pregame Model.")

if __name__ == "__main__":
    main()