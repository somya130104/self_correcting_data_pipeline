import json
import sqlite3
import os
import sys

# Define base directory relative to the script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "input", "reviews.db")
JSON_PATH = os.path.join(BASE_DIR, "input", "yelp_academic_dataset_review.json")

def seed_database(limit=None):
    print(f"Initializing SQLite database at: {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the table with strict schemas and our pipeline_status state column
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_reviews (
            review_id TEXT PRIMARY KEY,
            business_id TEXT,
            user_id TEXT,
            stars REAL,
            text TEXT,
            date TEXT,
            useful INTEGER,
            funny INTEGER,
            cool INTEGER,
            pipeline_status TEXT DEFAULT 'PENDING'
        )
    """)
    
    # Indexing pipeline_status dramatically accelerates batch query speeds
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON customer_reviews (pipeline_status)")
    conn.commit()
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: Raw JSON dataset not found at {JSON_PATH}")
        sys.exit(1)
        
    print("Seeding data from JSON file... (This may take a few moments)")
    
    insert_query = """
        INSERT OR IGNORE INTO customer_reviews 
        (review_id, business_id, user_id, stars, text, date, useful, funny, cool, pipeline_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
    """
    
    batch = []
    batch_size = 5000
    count = 0
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            batch.append((
                record.get("review_id"),
                record.get("business_id"),
                record.get("user_id"),
                float(record.get("stars", 0)),
                record.get("text"),
                record.get("date"),
                int(record.get("useful", 0)),
                int(record.get("funny", 0)),
                int(record.get("cool", 0))
            ))
            
            if len(batch) >= batch_size:
                cursor.executemany(insert_query, batch)
                conn.commit()
                count += len(batch)
                print(f"Inserted {count:,} records...")
                batch = []
                
            if limit and count >= limit:
                break
                
        if batch:
            cursor.executemany(insert_query, batch)
            conn.commit()
            count += len(batch)
            
    print(f"\nSuccess! Successfully seeded {count:,} records into SQLite.")
    conn.close()

if __name__ == "__main__":
    # You can pass an integer limit if you want to test with a smaller dataset first
    seed_database()
