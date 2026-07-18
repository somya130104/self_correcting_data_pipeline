import itertools
import json
import os
import re
from datetime import datetime, timedelta
import logging
import sqlite3
from typing import Any

# pyrefly: ignore [missing-import]
from airflow.sdk import dag, task, Param, get_current_context

# Initialize logger
logger = logging.getLogger(__name__)


class Config:
    BASE_DIR = os.getenv('PIPELINE_BASE_DIR', '/Users/somyabhadada/self_correcting_data_pipeline')
    SQLITE_DB = os.getenv('PIPELINE_DB_PATH', f'{BASE_DIR}/input/reviews.db')
    OUTPUT_DIR = os.getenv('PIPELINE_OUTPUT_DIR', f'{BASE_DIR}/output/')
    MAX_TEXT_LENGTH = 2000
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_OFFSET = 0
    
    OLLAMA_HOST = 'http://localhost:11434'
    OLLAMA_MODEL = 'llama3.2'
    OLLAMA_RETRIES = 3


# Default arguments for the DAG
default_args = {
    'owner': 'Somya Bhadada',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'execution_timeout': timedelta(minutes=30),
}


def _load_ollama_model(model_name: str):
    """Load and validate the Ollama model, pulling it if not already available."""
    import ollama

    client = ollama.Client(host=Config.OLLAMA_HOST)

    try:
        client.show(model_name)
        logger.info(f"Model '{model_name}' is already available.")
    except ollama.ResponseError:
        logger.info(f"Model '{model_name}' not found locally. Attempting to pull...")
        client.pull(model_name)
        logger.info(f"Model '{model_name}' pulled successfully.")

    # Test inference with a dummy prompt
    logger.info(f"Running test inference with model '{model_name}'...")
    response = client.chat(
        model=model_name,
        messages=[
            {
                'role': 'user',
                'content': "Classify the sentiment: 'This is a great product!' as positive, negative, or neutral."
            }
        ]
    )

    result = response['message']['content'].strip().upper()
    logger.info(f"Test inference result: {result}")

    return {
        'backend': 'ollama',
        'model_name': model_name,
        'ollama_host': Config.OLLAMA_HOST,
        'max_length': Config.MAX_TEXT_LENGTH,
        'status': 'loaded',
        'validated_at': datetime.now().isoformat(),
    }


def _load_from_sqlite(batch_size: int) -> list:
    """Queries SQLite for PENDING records, enforcing a clean decoupled schema entry."""
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
    return reviews


def _parse_ollama_response(response_text: str) -> dict:
    """Parse a sentiment result from an Ollama model response.

    The model may wrap its JSON answer in a markdown code-fence block.
    This function strips that formatting before attempting to parse the
    payload, and falls back to simple keyword detection if JSON parsing
    fails for any reason.

    Args:
        response_text: Raw string returned by the Ollama chat API.

    Returns:
        A dict with keys:
            'label' – one of 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'.
            'score' – float confidence in [0.0, 1.0].
    """
    VALID_SENTIMENTS = {'POSITIVE', 'NEGATIVE', 'NEUTRAL'}

    # --- Step 1: strip markdown code-fence formatting if present ----------
    text = response_text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        # Drop the opening fence (line 0) and the closing fence (last line).
        inner_lines = lines[1:-1]
        text = '\n'.join(inner_lines).strip()

    # --- Step 2: attempt structured JSON parsing -------------------------
    try:
        data = json.loads(text)

        sentiment = str(data.get('sentiment', 'NEUTRAL')).upper()
        if sentiment not in VALID_SENTIMENTS:
            sentiment = 'NEUTRAL'

        confidence = float(data.get('confidence', 0.0))
        confidence = max(0.0, min(1.0, confidence))

        return {'label': sentiment, 'score': confidence}

    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        # --- Step 3: keyword-based fallback ------------------------------
        upper_text = response_text.upper()
        if 'POSITIVE' in upper_text:
            return {'label': 'POSITIVE', 'score': 0.75}
        if 'NEGATIVE' in upper_text:
            return {'label': 'NEGATIVE', 'score': 0.75}
        return {'label': 'NEUTRAL', 'score': 0.5}


def _heal_review(review: dict) -> dict:
    """Apply self-healing rules to a raw review record.

    Inspects the 'text' field and corrects common data quality issues in order
    of severity.  Every healing action is recorded in the returned dict so that
    downstream tasks can audit what was changed.

    Healing rules (applied in order, first match wins):
        1. missing_text       – text is None
        2. wrong_type         – text is not a str (attempts cast)
        3. empty_text         – text is blank / whitespace-only
        4. special_characters_only – text contains no alphanumeric characters
        5. too_long           – text exceeds Config.MAX_TEXT_LENGTH characters

    Args:
        review: Raw review dict as produced by _load_from_file.

    Returns:
        A dict containing the healed text, provenance flags, and metadata.
    """
    PLACEHOLDER_MISSING = 'No review text provided.'
    PLACEHOLDER_NONCONTENT = '[Non-text content]'

    text: bool | float | int | str | None = review.get('text')

    # --- Base result dict ------------------------------------------------
    result: dict[str, bool | float | int | str | None | dict] = {
        'review_id':   review.get('review_id'),
        'business_id': review.get('business_id'),
        'stars':       review.get('stars'),
        'original_text': text if isinstance(text, (str, int, float, bool)) else None,
        'error_type':    None,
        'action_taken':  'none',
        'was_healed':    False,
        'metadata': {
            'user_id': review.get('user_id'),
            'date':    review.get('date'),
            'useful':  review.get('useful'),
            'funny':   review.get('funny'),
            'cool':    review.get('cool'),
        },
    }

    # --- Healing rule 1: missing text ------------------------------------
    if text is None:
        result['error_type']  = 'missing_text'
        result['action_taken'] = 'filled_with_placeholder'
        result['healed_text']  = PLACEHOLDER_MISSING
        result['was_healed']   = True

    # --- Healing rule 2: wrong type --------------------------------------
    elif not isinstance(text, str):
        result['error_type'] = 'wrong_type'
        result['was_healed'] = True
        try:
            result['healed_text']  = str(text)
            result['action_taken'] = 'type_conversion'
        except Exception:
            result['healed_text']  = 'Conversion failed.'
            result['action_taken'] = 'type_conversion'

    # --- Healing rule 3: empty / whitespace-only string ------------------
    elif not text.strip():
        result['error_type']   = 'empty_text'
        result['action_taken'] = 'filled_with_placeholder'
        result['healed_text']  = PLACEHOLDER_MISSING
        result['was_healed']   = True

    # --- Healing rule 4: special characters only -------------------------
    elif not re.search(r'[a-zA-Z0-9]', text):
        result['error_type']   = 'special_characters_only'
        result['action_taken'] = 'replaced_with_placeholder'
        result['healed_text']  = PLACEHOLDER_NONCONTENT
        result['was_healed']   = True

    # --- Healing rule 5: text exceeds maximum length ---------------------
    elif len(text) > Config.MAX_TEXT_LENGTH:
        result['error_type']   = 'too_long'
        result['action_taken'] = 'truncated_text'
        result['healed_text']  = text[:Config.MAX_TEXT_LENGTH] + '...'
        result['was_healed']   = True

    # --- No issues detected ----------------------------------------------
    else:
        result['healed_text'] = text.strip()
        result['was_healed']  = False

    return result


def _created_degraded_results(healed_reviews: list[dict], error_message: str) -> list[dict]:
    """Build a degraded result list when the inference backend is unavailable.

    Every review receives a neutral prediction with a 0.5 confidence score and
    a 'degraded' status so that downstream tasks can distinguish genuine
    predictions from fallbacks.

    Args:
        healed_reviews: Output of the healing stage (list of healed review dicts).
        error_message:  Human-readable reason why inference could not run.

    Returns:
        A list of result dicts, one per input review, all marked as degraded.
    """
    return [
        {
            'review_id':          review.get('review_id'),
            'business_id':        review.get('business_id'),
            'stars':              review.get('stars'),
            'healed_text':        review.get('healed_text'),
            'healing_applied':    review.get('was_healed', False),
            'error_type':         review.get('error_type'),
            'predicted_sentiment': 'NEUTRAL',
            'confidence':          0.5,
            'status':             'degraded',
            'error_message':      error_message,
            'metadata':           review.get('metadata', {}),
        }
        for review in healed_reviews
    ]


def _analyze_with_ollama(healed_reviews: list[dict], model_info: dict) -> list[dict]:
    """Run sentiment inference on healed reviews using a local Ollama model.

    For each review the function sends a structured prompt to the Ollama chat
    API and parses the response with _parse_ollama_response. Failed calls are
    retried up to Config.OLLAMA_RETRIES times with a 1-second back-off before
    the review is assigned a neutral fallback prediction.

    Uses `asyncio` and `ollama.AsyncClient` with a semaphore limit of 4 to leverage
    continuous batching and concurrent processing without exhausting local resources.

    If the Ollama client itself cannot be initialised (e.g. the server is not
    running), the entire batch is handed off to _created_degraded_results
    immediately.

    Args:
        healed_reviews: List of healed review dicts from the healing stage.
        model_info:     Dict returned by _load_ollama_model (contains
                        'model_name', 'ollama_host', etc.).

    Returns:
        A list of result dicts, one per input review, with prediction fields.
    """
    import asyncio
    from ollama import AsyncClient
    
    model_name  = model_info.get('model_name',  Config.OLLAMA_MODEL)
    ollama_host = model_info.get('ollama_host', Config.OLLAMA_HOST)
    total       = len(healed_reviews)

    async def _run_batch():
        # --- Initialise the Ollama client ------------------------------------
        try:
            client = AsyncClient(host=ollama_host)
            # Lightweight connectivity check – will raise if server is unreachable.
            await client.list()
        except Exception as exc:
            error_msg = f"Failed to connect to Ollama at {ollama_host}: {exc}"
            logger.error(error_msg)
            return _created_degraded_results(healed_reviews, error_msg)

        # Set a strict limit of 4 concurrent requests to prevent choking the system
        semaphore = asyncio.Semaphore(4)
        
        async def infer_single(review, idx):
            text_to_classify = review.get('healed_text', '')

            prompt = (
                f"Classify the sentiment of the following review as POSITIVE, NEGATIVE, or NEUTRAL.\n"
                f"Return ONLY a JSON object in this exact format, with no other text:\n"
                f'{{ "sentiment": "POSITIVE|NEGATIVE|NEUTRAL", "confidence": 0.0-1.0 }}\n\n'
                f"Review:\n{text_to_classify}"
            )

            prediction  = None
            last_error  = None

            async with semaphore:
                # --- Retry loop --------------------------------------------------
                for attempt in range(Config.OLLAMA_RETRIES):
                    try:
                        response = await client.chat(
                            model=model_name,
                            messages=[{'role': 'user', 'content': prompt}],
                            options={'temperature': 0.1},
                        )
                        response_text = response['message']['content']
                        prediction    = _parse_ollama_response(response_text)
                        break  # success – exit retry loop
                    except Exception as exc:
                        last_error = exc
                        logger.warning(
                            "Ollama inference attempt %d/%d failed for review '%s': %s",
                            attempt + 1,
                            Config.OLLAMA_RETRIES,
                            review.get('review_id'),
                            exc,
                        )
                        await asyncio.sleep(1)

            # All retries exhausted – fall back to neutral prediction.
            if prediction is None:
                logger.error(
                    "All %d retries failed for review '%s'. Assigning neutral fallback. Last error: %s",
                    Config.OLLAMA_RETRIES,
                    review.get('review_id'),
                    last_error,
                )
                prediction = {'label': 'NEUTRAL', 'score': 0.5}

            # Log progress every 10 reviews.
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                logger.info(
                    "Inference progress: %d / %d reviews processed.",
                    idx + 1,
                    total,
                )

            return {
                'review_id':           review.get('review_id'),
                'business_id':         review.get('business_id'),
                'stars':               review.get('stars'),
                'healed_text':         text_to_classify,
                'healing_applied':     review.get('was_healed', False),
                'error_type':          review.get('error_type'),
                'predicted_sentiment': prediction['label'],
                'confidence':          prediction['score'],
                'status':              'healed' if review.get('was_healed') else 'success',
                'error_message':       None,
                'metadata':            review.get('metadata', {}),
            }

        # Construct and launch all tasks concurrently
        tasks = [infer_single(review, idx) for idx, review in enumerate(healed_reviews)]
        results = await asyncio.gather(*tasks)
        return list(results)

    # Bridge synchronous Airflow to async pipeline
    return asyncio.run(_run_batch())


def _write_state_to_sqlite(results: list):
    """Commits pipeline processing updates back to the immutable datastore."""
    if not results:
        return
        
    conn = sqlite3.connect(Config.SQLITE_DB)
    cursor = conn.cursor()
    
    update_query = """
        UPDATE customer_reviews 
        SET pipeline_status = ?, text = ? 
        WHERE review_id = ?
    """
    
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
    logger.info(f"Committed state transitions for {len(results)} records to SQLite.")


def _aggregate_results(results: list[dict], params: dict) -> dict:
    """Aggregate inference results, persist a summary JSON, and return metrics.

    Counts success, healing, and degraded reviews; computes rates; builds
    distribution tables for sentiment, healing actions, and star-sentiment
    correlation; writes a timestamped JSON file to Config.OUTPUT_DIR; and
    returns all summary fields *except* the raw results list (to keep the
    XCom payload small).

    Args:
        results: List of result dicts produced by _analyze_with_ollama.
        params:  DAG-run params dict (used for run_info fields).

    Returns:
        A summary dict with run_info, totals, rates, sentiment_distribution,
        healing_statistics, and star_sentiment_correlation.
    """
    total          = len(results)
    success_count  = sum(1 for r in results if r.get('status') == 'success')
    healed_count   = sum(1 for r in results if r.get('status') == 'healed')
    degraded_count = sum(1 for r in results if r.get('status') == 'degraded')

    # --- Rates (guard against empty batch) ----------------------------------
    success_rate     = round(success_count  / total, 4) if total else 0.0
    healing_rate     = round(healed_count   / total, 4) if total else 0.0
    degradation_rate = round(degraded_count / total, 4) if total else 0.0

    # --- Sentiment distribution ---------------------------------------------
    sentiment_dist: dict[str, int] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    for r in results:
        label = str(r.get('predicted_sentiment', 'NEUTRAL')).upper()
        if label in sentiment_dist:
            sentiment_dist[label] += 1

    # --- Healing action breakdown -------------------------------------------
    healing_stats: dict[str, int] = {}
    for r in results:
        action = r.get('action_taken', 'none')
        if action and action != 'none':
            healing_stats[action] = healing_stats.get(action, 0) + 1

    # --- Star-sentiment correlation -----------------------------------------
    star_sentiment: dict[str, dict[str, int]] = {}
    for r in results:
        star_key  = str(r.get('stars', 'unknown'))
        sentiment = str(r.get('predicted_sentiment', 'NEUTRAL')).upper()
        if star_key not in star_sentiment:
            star_sentiment[star_key] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        if sentiment in star_sentiment[star_key]:
            star_sentiment[star_key][sentiment] += 1

    # --- Build full summary -------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    raw_offset = params.get('offset')
    offset = int(raw_offset) if raw_offset is not None else Config.DEFAULT_OFFSET
    
    raw_batch_size = params.get('batch_size')
    batch_size = int(raw_batch_size) if raw_batch_size is not None else Config.DEFAULT_BATCH_SIZE
    
    raw_model = params.get('model_name')
    model_name = str(raw_model) if raw_model is not None else Config.OLLAMA_MODEL
    
    raw_db = params.get('sqlite_db_path')
    sqlite_db_path = str(raw_db) if raw_db is not None else Config.SQLITE_DB

    summary = {
        'run_info': {
            'timestamp':  timestamp,
            'offset':     offset,
            'batch_size': batch_size,
            'model':      model_name,
            'sqlite_db_path': sqlite_db_path,
        },
        'totals': {
            'total':     total,
            'success':   success_count,
            'healed':    healed_count,
            'degraded':  degraded_count,
        },
        'rates': {
            'success_rate':     success_rate,
            'healing_rate':     healing_rate,
            'degradation_rate': degradation_rate,
        },
        'sentiment_distribution':   sentiment_dist,
        'healing_statistics':       healing_stats,
        'star_sentiment_correlation': star_sentiment,
        'results':                  results,
    }

    # --- Persist to disk ----------------------------------------------------
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_filename = f'sentiment_analysis_summary_{timestamp}_Offset{offset}.json'
    output_path     = os.path.join(Config.OUTPUT_DIR, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Summary written to %s", output_path)

    # Return summary without the raw results list to keep XCom payload light.
    summary_without_results = {k: v for k, v in summary.items() if k != 'results'}
    return summary_without_results


def _generate_health_report(summary: dict) -> dict:
    """Derive a pipeline health report from an aggregated summary.

    Classifies pipeline health into four tiers based on degraded and healed
    review proportions and returns a structured report dict suitable for
    logging, alerting, or downstream consumption.

    Health tiers (evaluated in order):
        CRITICAL  – more than 10 % of reviews are degraded.
        DEGRADED  – at least one review is degraded.
        WARNING   – more than 50 % of reviews required healing.
        HEALTHY   – no issues detected.

    Args:
        summary: Dict returned by _aggregate_results (without raw results).

    Returns:
        A health report dict with pipeline, timestamp, health_status,
        run_info, metrics, sentiment_distribution, and healing_summary keys.
    """
    totals   = summary.get('totals', {})
    total    = int(totals.get('total',    0))
    healed   = int(totals.get('healed',   0))
    degraded = int(totals.get('degraded', 0))

    # --- Determine health tier ----------------------------------------------
    if total > 0 and degraded > total * 0.10:
        health_status = 'CRITICAL'
    elif degraded > 0:
        health_status = 'DEGRADED'
    elif total > 0 and healed > total * 0.50:
        health_status = 'WARNING'
    else:
        health_status = 'HEALTHY'

    logger.info("Pipeline health status: %s", health_status)

    run_info = summary.get('run_info', {})

    return {
        'pipeline':              'self_healing_sentiment_pipeline',
        'timestamp':             run_info.get('timestamp', datetime.now().isoformat()),
        'health_status':         health_status,
        'run_info':              run_info,
        'metrics': {
            'totals': totals,
            'rates':  summary.get('rates', {}),
        },
        'sentiment_distribution': summary.get('sentiment_distribution', {}),
        'healing_summary': {
            'healed_count':  healed,
            'healing_rate':  summary.get('rates', {}).get('healing_rate', 0.0),
            'healing_stats': summary.get('healing_statistics', {}),
        },
    }


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

@dag(
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
    def heal_reviews(raw_reviews: Any) -> list:
        return [_heal_review(review) for review in raw_reviews]

    @task()
    def analyze_sentiment(healed_reviews: Any, model_info: Any) -> list:
        return _analyze_with_ollama(healed_reviews, model_info)

    @task()
    def update_database(results: Any):
        _write_state_to_sqlite(results)

    @task()
    def aggregate(results: Any, **context) -> dict:
        return _aggregate_results(results, context['params'])

    @task()
    def health_report(summary: Any) -> dict:
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
    _ = inference >> commit_db

self_healing_pipeline()
