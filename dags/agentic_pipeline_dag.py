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
