import subprocess
import os
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def trigger_dag_run(batch_size: int, dry_run: bool = False) -> dict:
    """Triggers an Airflow batch consumer worker to claim pending records."""
    conf = json.dumps({
        "batch_size": batch_size,
        "data_source": "sqlite",
        "model_backend": "ollama"
    })

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    airflow_path = os.path.join(project_root, 'venv', 'bin', 'airflow')
    
    cmd = [
        airflow_path, "dags", "trigger",
        "self_healing_sentiment_pipeline",
        "--conf", conf
    ]

    if dry_run:
        print(f"[DRY RUN] Would trigger consumer worker for batch_size={batch_size}")
        return {"status": "dry_run"}

    env = os.environ.copy()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env['AIRFLOW_HOME'] = project_root

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"Triggered parallel worker consuming batch size: {batch_size}")
            return {"status": "triggered"}
        else:
            print(f"Failed to boot worker: {result.stderr}")
            return {"status": "failed", "error": result.stderr}
    except Exception as e:
        print(f"Execution Error: {e}")
        return {"status": "error", "error": str(e)}

def run_batch_processing(total_records: int, batch_size: int, parallel: int = 1, dry_run: bool = False):
    total_batches = total_records // batch_size
    
    print(f"\n{'='*60}")
    print(f"Concurrent Database Processing Worker Node Configuration")
    print(f"{'='*60}")
    print(f"Total target records:  {total_records:,}")
    print(f"Records per batch:     {batch_size:,}")
    print(f"Total spawned workers: {total_batches:,}")
    print(f"Parallel consumers:    {parallel}")
    print(f"{'='*60}\n")

    results = {"triggered": 0, "failed": 0, "dry_run": 0}

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [
            executor.submit(trigger_dag_run, batch_size, dry_run)
            for _ in range(total_batches)
        ]

        for future in as_completed(futures):
            res = future.result()
            results[res["status"]] = results.get(res["status"], 0) + 1

    print(f"\n{'='*60}")
    print(f"Execution Architecture Summary: {results}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Stateful Queue Batch Runner")
    parser.add_argument("--total", type=int, required=True, help="Total records to process")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size per worker")
    parser.add_argument("--parallel", type=int, default=1, help="Max concurrent consumers")
    parser.add_argument("--dry-run", action="store_true", help="Perform configuration dry run")

    args = parser.parse_args()
    run_batch_processing(args.total, args.batch_size, args.parallel, args.dry_run)

if __name__ == "__main__":
    main()
