import sys
import re

with open("dags/agentic_pipeline_dag.py", "r") as f:
    content = f.read()

# Add sqlite3 import if not present
if "import sqlite3" not in content:
    content = content.replace("import itertools\n", "import itertools\nimport sqlite3\n")

# Replace Config
new_config = """class Config:
    BASE_DIR = os.getenv('PIPELINE_BASE_DIR', '/Users/somyabhadada/self_correcting_data_pipeline')
    SQLITE_DB = os.getenv('PIPELINE_DB_PATH', f'{BASE_DIR}/input/reviews.db')
    OUTPUT_DIR = os.getenv('PIPELINE_OUTPUT_DIR', f'{BASE_DIR}/output/')
    MAX_TEXT_LENGTH = 2000
    DEFAULT_BATCH_SIZE = 100
    
    OLLAMA_HOST = 'http://localhost:11434'
    OLLAMA_MODEL = 'llama3.2'
    OLLAMA_RETRIES = 3"""

content = re.sub(r'class Config:.*?OLLAMA_RETRIES = 3\n', new_config + "\n", content, flags=re.DOTALL)

# Replace _load_from_sqlite
new_load = """def _load_from_sqlite(batch_size: int) -> list:
    \"\"\"Queries SQLite for PENDING records, enforcing a clean decoupled schema entry.\"\"\"
    if not os.path.exists(Config.SQLITE_DB):
        raise FileNotFoundError(f"Database file not found at: {Config.SQLITE_DB}")
        
    conn = sqlite3.connect(Config.SQLITE_DB)
    conn.row_factory = sqlite3.Row  # Enables fetching columns by dictionary keys
    cursor = conn.cursor()
    
    # Select rows with a strict transaction limit
    cursor.execute(
        "SELECT * FROM customer_reviews WHERE pipeline_status = 'PENDING' LIMIT ?", 
        (batch_size,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    reviews = []
    for row in rows:
        reviews.append({
            'review_id':   row['review_id'],
            'business_id': row['business_id'],
            'user_id':     row['user_id'],
            'stars':       row['stars'],
            'text':        row['text'],
            'date':        row['date'],
            'useful':      row['useful'],
            'funny':       row['funny'],
            'cool':        row['cool'],
        })
    return reviews"""

content = re.sub(r'def _load_from_sqlite\(.*?\).*?return reviews\n', new_load + "\n", content, flags=re.DOTALL)

# Replace _update_db_status with _write_state_to_sqlite
new_write = """def _write_state_to_sqlite(results: list):
    \"\"\"Commits pipeline processing updates back to the immutable datastore.\"\"\"
    if not results:
        return
        
    conn = sqlite3.connect(Config.SQLITE_DB)
    cursor = conn.cursor()
    
    update_query = \"\"\"
        UPDATE customer_reviews 
        SET pipeline_status = ?, text = ? 
        WHERE review_id = ?
    \"\"\"
    
    update_batch = []
    for r in results:
        # Map statuses cleanly into the database schema
        status = r.get('status', '').upper()  # 'SUCCESS', 'HEALED', or 'DEGRADED'
        healed_text = r.get('healed_text')
        review_id = r.get('review_id')
        update_batch.append((status, healed_text, review_id))
        
    cursor.executemany(update_query, update_batch)
    conn.commit()
    conn.close()
    logger.info(f"Committed state transitions for {len(results)} records to SQLite.")"""

content = re.sub(r'def _update_db_status\(.*?\).*?len\(results\)\)\n', new_write + "\n", content, flags=re.DOTALL)

# Replace dag definition
new_dag = """@dag(
    dag_id='self_healing_sentiment_pipeline',
    default_args=default_args,
    schedule=None,
    params={
        'batch_size': Param(Config.DEFAULT_BATCH_SIZE, type='integer', minimum=1),
        'model_name': Param(Config.OLLAMA_MODEL, type='string'),
    },
    tags=['sentiment', 'self-healing', 'sqlite'],
)
def self_healing_pipeline():

    @task()
    def load_model(**context) -> dict:
        return _load_ollama_model(str(context['params'].get('model_name', Config.OLLAMA_MODEL)))

    @task()
    def ingest_data(**context) -> list:
        batch_size = int(context['params'].get('batch_size', Config.DEFAULT_BATCH_SIZE))
        return _load_from_sqlite(batch_size)

    @task()
    def heal_reviews(raw_reviews: list) -> list:
        return [_heal_review(review) for review in raw_reviews]

    @task()
    def analyze_sentiment(healed_reviews: list, model_info: dict) -> list:
        return _analyze_with_ollama(healed_reviews, model_info)

    @task()
    def update_database(results: list):
        _write_state_to_sqlite(results)

    @task()
    def aggregate(results: list, **context) -> dict:
        return _aggregate_results(results, context['params'])

    @task()
    def health_report(summary: dict) -> dict:
        return _generate_health_report(summary)

    # Task Dependency Graph Wire-up
    model_info  = load_model()
    raw_data    = ingest_data()
    healed      = heal_reviews(raw_data)
    inference   = analyze_sentiment(healed, model_info)
    
    # State tracking runs concurrently alongside local metric aggregations
    commit_db   = update_database(inference)
    summary     = aggregate(inference)
    report      = health_report(summary)
    
    # Ensure database writes complete before closing tasks
    inference >> commit_db

self_healing_pipeline()"""

content = re.sub(r'@dag\(\n    dag_id=\'self_healing_sentiment_pipeline\'.*', new_dag + "\n", content, flags=re.DOTALL)

with open("dags/agentic_pipeline_dag.py", "w") as f:
    f.write(content)
