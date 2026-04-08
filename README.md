# DataCleaningEnv

Production-ready OpenEnv environment for data cleaning RL tasks with FastAPI.

## Project Structure

```
DataCleaningEnv/
  server/
    app.py
    env.py
    models.py
    tasks/
      easy.py
      medium.py
      hard.py
    graders/
      df_grader.py
    data/
      easy.csv
      easy_clean.csv
      medium.csv
      medium_clean.csv
      hard.csv
      hard_clean.csv
    Dockerfile
  inference.py
  openenv.yaml
  README.md
```

## Environment API

- `GET /reset` with optional query `?task_id=easy|medium|hard`
- `POST /reset` with `{ "task_id": "easy|medium|hard" }` (compatibility alias)
- `POST /step` with action payload
- `GET /state` (and `POST /state` alias)

### Action Space

- `drop_nulls`
- `fill_nulls`
- `drop_duplicates`
- `cast_column`
- `normalize`
- `clip_outliers`
- `submit`

### Observation Fields

- `shape`
- `columns`
- `dtypes`
- `null_counts`
- `sample_rows`
- `duplicate_count`
- `step_number`
- `last_error`

### Reward

- `+0.01` for valid action
- `-0.05` for invalid action
- On `submit`: F1 score vs ground truth in `[0, 1]`

## Local Run

1. Install deps:

```bash
pip install fastapi uvicorn pandas pydantic openai
```

2. Start server:

```bash
cd server
uvicorn app:app --host 0.0.0.0 --port 8000
```

3. Run inference:

```bash
cd ..
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your_token
set ENV_BASE_URL=http://localhost:8000
set TASK_ID=easy
py inference.py
```

Expected log prefixes from inference:

- `[START]`
- `[STEP]`
- `[END]`

## Docker

Build and run:

```bash
docker build -t data-cleaning-env .
docker run -p 8000:8000 data-cleaning-env
```

## OpenEnv Validation

If `openenv` validator is available in your environment:

```bash
openenv validate openenv.yaml
```

If the global `openenv` command is unavailable, run the built-in strict validator:

```bash
py openenv_validate.py
```

Expected output:

```text
openenv validate passed
```

The environment follows the required reset/step/state contract and typed Pydantic payloads.
