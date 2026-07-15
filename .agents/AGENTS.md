# Workspace Rules

## Python Interpreter
- Always use the project-local virtual environment at `./venv/bin/python` for this workspace.
- The Python interpreter path is: `/Users/somyabhadada/Desktop/self_correcting_data_pipeline/venv/bin/python`
- Python version: 3.12.3
- Airflow version: 3.3.0 (installed in venv)
- Do NOT use the system/miniconda Python (`/Users/somyabhadada/miniconda3/lib/python3.13/`) for this project.

## Running Commands
- Always activate the venv before running Python commands: `source venv/bin/activate`
- Use `./venv/bin/airflow` or activate the venv to run Airflow CLI commands.
- To start Airflow: `source venv/bin/activate && airflow standalone`
