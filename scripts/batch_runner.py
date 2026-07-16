import subprocess
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def trigger_dag_run(offset: int, batch_size: int, dry_run: bool = False) -> dict:
    """Trigger a single DAG run with the given offset."""
    conf = json.dumps({
        "batch_size": batch_size,
        "offset": offset,
        "data_source": "file",
        "model_backend": "ollama"
    })

    cmd = [
        "airflow", "dags", "trigger",
        "self_healing_sentiment_pipeline",
        "--conf", conf
    ]

    if dry_run:
        print(f"[DRY RUN] Would trigger: offset={offset}, batch_size={batch_size}")
        return {"offset": offset, "status": "dry_run"}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"Triggered: offset={offset:,} - {offset + batch_size:,}")
            return {"offset": offset, "status": "triggered"}
        else:
            print(f"Failed: offset={offset} - {result.stderr}")
            return {"offset": offset, "status": "failed", "error": result.stderr}
    except Exception as e:
        print(f"Error: offset={offset} - {e}")
        return {"offset": offset, "status": "error", "error": str(e)}


def run_batch_processing(
    total_records: int,
    batch_size: int,
    parallel: int = 1,
    start_offset: int = 0,
    delay: float = 1.0,
    dry_run: bool = False
):
    """Run batch processing with optional parallelism."""

    offsets = list(range(start_offset, total_records, batch_size))
    total_batches = len(offsets)

    print(f"\n{'='*60}")
    print(f"Batch Processing Configuration")
    print(f"{'='*60}")
    print(f"Total records:  {total_records:,}")
    print(f"Batch size:     {batch_size:,}")
    print(f"Total batches:  {total_batches:,}")
    print(f"Parallelism:    {parallel}")
    print(f"Start offset:   {start_offset:,}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN MODE - No actual triggers]\n")

    results = {"triggered": 0, "failed": 0}

    if parallel == 1:
        # Sequential processing
        for i, offset in enumerate(offsets):
            result = trigger_dag_run(offset, batch_size, dry_run)
            results[result["status"]] = results.get(result["status"], 0) + 1

            if not dry_run and delay > 0 and i < len(offsets) - 1:
                time.sleep(delay)
    else:
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(trigger_dag_run, offset, batch_size, dry_run): offset
                for offset in offsets
            }

            for future in as_completed(futures):
                result = future.result()
                results[result["status"]] = results.get(result["status"], 0) + 1

    print(f"\n{'='*60}")
    print(f"Summary: {results}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for sentiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Process 5M records with batch size 1000 (sequential)
            python scripts/batch_runner.py --total 5000000 --batch-size 1000

            # Process with 5 parallel DAG runs
            python scripts/batch_runner.py --total 5000000 --batch-size 5000 --parallel 5

            # Resume from offset 100000
            python scripts/batch_runner.py --total 5000000 --batch-size 1000 --start 100000

            # Dry run to see what would be triggered
            python scripts/batch_runner.py --total 5000000 --batch-size 10000 --dry-run
        """
    )

    parser.add_argument("--total", type=int, required=True,
                        help="Total number of records to process")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Records per DAG run (default: 1000)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel DAG runs to trigger (default: 1)")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting offset for resume (default: 0)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between triggers in seconds (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be triggered without executing")

    args = parser.parse_args()

    run_batch_processing(
        total_records=args.total,
        batch_size=args.batch_size,
        parallel=args.parallel,
        start_offset=args.start,
        delay=args.delay,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
