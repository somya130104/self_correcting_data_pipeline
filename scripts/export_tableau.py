import sqlite3
import csv
import os

def export_to_csv():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    db_path = os.path.join(project_root, 'input', 'reviews.db')
    output_dir = os.path.join(project_root, 'output')
    csv_path = os.path.join(output_dir, 'tableau_export.csv')

    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # We only export rows that have been processed to keep the CSV lightweight and fast.
    query = "SELECT * FROM customer_reviews WHERE pipeline_status != 'PENDING';"
    cursor.execute(query)
    
    rows = cursor.fetchall()
    if not rows:
        print("No processed records found to export.")
        conn.close()
        return

    # Extract column names from the cursor description
    col_names = [description[0] for description in cursor.description]

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting {len(rows)} records to CSV...")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(col_names)  # Write header
        writer.writerows(rows)      # Write data
        
    conn.close()
    print(f"Success! Data exported to {csv_path}")

if __name__ == "__main__":
    export_to_csv()
