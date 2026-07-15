import itertools
import json
import os
import re
from datetime import datetime, timedelta
import logging

# pyrefly: ignore [missing-import]
from airflow.sdk import dag, task, Param, get_current_context

# Initialize logger
logger = logging.getLogger(__name__)


class Config:
    """Pipeline configuration class."""

    BASE_DIR = os.getenv(
        'PIPELINE_BASE_DIR',
        '/Users/somyabhadada/Desktop/self_correcting_data_pipeline'
    )
    INPUT_FILE = os.getenv(
        'PIPELINE_INPUT_FILE',
        f'{BASE_DIR}/input/yelp_academic_dataset_review.json'
    )
    OUTPUT_DIR = os.getenv(
        'PIPELINE_OUTPUT_DIR',
        f'{BASE_DIR}/output/'
    )
    MAX_TEXT_LENGTH = 2000
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_OFFSET = 0

    # Ollama variables
    OLLAMA_HOST = 'http://localhost:11434'
    OLLAMA_MODEL = 'llama3.2'
    OLLAMA_TIMEOUT = 120
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


def _load_from_file(params: dict, batch_size: int, offset: int) -> list:
    """Read a batch of reviews from a JSON-lines input file.

    Uses itertools.islice to efficiently seek to the desired offset without
    loading the entire file into memory, then parses each line as a JSON object.
    Malformed lines are logged as warnings and skipped gracefully.

    Args:
        params:     DAG run params dict; may contain an 'input_file' override.
        batch_size: Number of lines to read in this batch.
        offset:     Zero-based line index at which the batch starts.

    Returns:
        A list of review dicts extracted from the batch.

    Raises:
        FileNotFoundError: If the resolved input file path does not exist.
    """
    input_file = params.get('input_file', Config.INPUT_FILE)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    reviews = []

    with open(input_file, encoding='utf-8') as f:
        batch = itertools.islice(f, offset, offset + batch_size)
        for line in batch:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping invalid JSON line at approximately offset %d: %s",
                    offset + len(reviews),
                    exc,
                )
                continue

            reviews.append({
                'review_id':   record.get('review_id'),
                'business_id': record.get('business_id'),
                'user_id':     record.get('user_id'),
                'stars':       record.get('stars', 0),
                'text':        record.get('text'),
                'date':        record.get('date'),
                'useful':      record.get('useful', 0),
                'funny':       record.get('funny', 0),
                'cool':        record.get('cool', 0),
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
    API and parses the response with _parse_ollama_response.  Failed calls are
    retried up to Config.OLLAMA_RETRIES times with a 1-second back-off before
    the review is assigned a neutral fallback prediction.

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
    import ollama
    import time

    model_name  = model_info.get('model_name',  Config.OLLAMA_MODEL)
    ollama_host = model_info.get('ollama_host', Config.OLLAMA_HOST)

    # --- Initialise the Ollama client ------------------------------------
    try:
        client = ollama.Client(host=ollama_host)
        # Lightweight connectivity check – will raise if server is unreachable.
        client.list()
    except Exception as exc:
        error_msg = f"Failed to connect to Ollama at {ollama_host}: {exc}"
        logger.error(error_msg)
        return _created_degraded_results(healed_reviews, error_msg)

    # --- Inference loop --------------------------------------------------
    results = []
    total   = len(healed_reviews)

    for idx, review in enumerate(healed_reviews):
        text_to_classify = review.get('healed_text', '')

        prompt = (
            f"Classify the sentiment of the following review as POSITIVE, NEGATIVE, or NEUTRAL.\n"
            f"Return ONLY a JSON object in this exact format, with no other text:\n"
            f'{{ "sentiment": "POSITIVE|NEGATIVE|NEUTRAL", "confidence": 0.0-1.0 }}\n\n'
            f"Review:\n{text_to_classify}"
        )

        prediction  = None
        last_error  = None

        # --- Retry loop --------------------------------------------------
        for attempt in range(Config.OLLAMA_RETRIES):
            try:
                response = client.chat(
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
                time.sleep(1)

        # All retries exhausted – fall back to neutral prediction.
        if prediction is None:
            logger.error(
                "All %d retries failed for review '%s'. Assigning neutral fallback. Last error: %s",
                Config.OLLAMA_RETRIES,
                review.get('review_id'),
                last_error,
            )
            prediction = {'label': 'NEUTRAL', 'score': 0.5}

        results.append({
            'review_id':           review.get('review_id'),
            'business_id':         review.get('business_id'),
            'stars':               review.get('stars'),
            'healed_text':         text_to_classify,
            'healing_applied':     review.get('was_healed', False),
            'error_type':          review.get('error_type'),
            'predicted_sentiment': prediction['label'],
            'confidence':          prediction['score'],
            'status':              'success',
            'error_message':       None,
            'metadata':            review.get('metadata', {}),
        })

        # Log progress every 10 reviews.
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            logger.info(
                "Inference progress: %d / %d reviews processed.",
                idx + 1,
                total,
            )

    return results
