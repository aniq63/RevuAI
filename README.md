<p align="center">
	<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
	<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
	<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn" />
	<img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" />
	<img src="https://img.shields.io/badge/Supabase-3FCF8E?style=for-the-badge&logo=supabase&logoColor=white" alt="Supabase" />
	<img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" alt="GitHub Actions" />
</p>

<h1 align="center">Revu AI</h1>

<p align="center">
<b>RevuAI is a Google Play Store review intelligence platform. It combines a Chrome extension with a FastAPI backend to collect app reviews, classify sentiment, identify recurring topics, and generate product insights with an LLM.</b>
</p>

<details>
<summary><strong>Table of Contents</strong></summary>

- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Run the API](#run-the-api)
- [API Reference](#api-reference)
	- [GET /](#get-)
	- [GET /health](#get-health)
	- [POST /inference/analyze](#post-inferenceanalyze)
	- [DELETE /inference/cache/{app_id}](#delete-inferencecacheapp_id)
- [Chrome Extension](#chrome-extension)
- [Data and ML Pipelines](#data-and-ml-pipelines)
	- [ETL from the labeled CSV](#etl-from-the-labeled-csv)
	- [Production-style ETL](#production-style-etl)
	- [Train and register a model](#train-and-register-a-model)
	- [Run inference directly](#run-inference-directly)
- [Docker](#docker)
- [Testing and Development](#testing-and-development)
- [Automation](#automation)
	- [Continuous integration and deployment](#continuous-integration-and-deployment)
	- [Monthly data and retraining cycle](#monthly-data-and-retraining-cycle)
	- [Workflow summary](#workflow-summary)
	- [Required repository secrets](#required-repository-secrets)
- [Repository Layout](#repository-layout)
- [Operational Notes](#operational-notes)
- [License](#license)

</details>

## Capabilities

- Analyze newest, most relevant, or all available reviews for a Play Store app.
- Classify reviews as positive, neutral, or negative with a registered Linear SVM model.
- Report sentiment percentages and the most helpful reviews in each sentiment group.
- Cluster review topics with MiniLM embeddings, UMAP, HDBSCAN, and KeyBERT.
- Generate a readable summary and recommendations with Groq.
- Cache recent analyses to make repeated extension requests faster.
- Maintain an append only review data lake in PostgreSQL/Supabase for model retraining.
- Track experiments and register Champion models through MLflow and DagsHub.

## Architecture

```mermaid
flowchart LR
	Extension[Chrome extension] --> API[FastAPI API]
	API --> Scraper[Google Play scraper]
	API --> Sentiment[Champion sentiment model]
	API --> Topics[Topic clustering]
	API --> LLM[Groq LLM insight]

	subgraph DataOps[Data and model lifecycle]
		ETL[Scheduled production ETL] --> Data[(PostgreSQL / Supabase)]
		Data --> Training[Scheduled monthly retraining]
		Training --> Evaluation[Evaluate against Champion]
		Evaluation --> Registry[MLflow / DagsHub registry]
		Registry --> Sentiment
	end

	subgraph Delivery[CI/CD delivery]
		Push[Push to main] --> Tests[Pytest]
		Tests --> Build[Build Docker image]
		Build --> GHCR[Push to GitHub Container Registry]
		GHCR --> Deploy[Deploy to EC2]
		Deploy --> Health[Health check]
	end

	ETL -. monthly schedule .-> Training
	Evaluation -- Champion promoted --> Build
```

The inference request runs through these stages:

1. Scrape reviews from Google Play.
2. Predict sentiment with the registered Champion model and TF-IDF vectorizer.
3. Calculate the sentiment distribution and select top reviews by helpfulness.
4. Cluster reviews by sentiment and extract representative topics.
5. Send the summarized results to Groq for a product-oriented insight.

## Requirements

- Python 3.11
- A PostgreSQL-compatible database for ETL and training data
- A Groq API key for LLM insight generation
- Access to the MLflow/DagsHub model registry for production inference
- Docker, if running the containerized service
- Google Chrome, if using the extension

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the full development dependency set:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the production API only, use `requirements-runtime.txt` instead. The Docker image installs the CPU-only PyTorch build and the runtime dependencies automatically.

## Configuration

Copy `.env.example` to `.env` and replace the placeholders:

```powershell
Copy-Item .env.example .env
```

Required values:

```dotenv
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST:5432/DATABASE
GROQ_API_KEY=your_groq_api_key
DAGSHUB_USERNAME=your_dagshub_username
DAGSHUB_USER_TOKEN=your_dagshub_token
```

`DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` are also supported by the scheduled ML workflow. Keep `.env` out of version control and never commit API keys or database credentials.

Optional runtime settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8000` | Port used by `python main.py` |
| `TORCH_NUM_THREADS` | `4` | CPU threads used by PyTorch |
| `RESULT_CACHE_TTL_SECONDS` | `1800` | In-memory inference cache lifetime |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model used for insight generation |

The ML pipeline reads model and training settings from [config/settings.yml](config/settings.yml). The database schema and connection setup are defined in [database/models.py](database/models.py) and [database/connection.py](database/connection.py).

## Run the API

Start the service with Uvicorn:

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Or run the application module directly:

```powershell
python main.py
```

The API loads the Champion sentiment model and the `all-MiniLM-L6-v2` sentence embedding model during startup. The first startup may download model artifacts and requires access to the configured MLflow/DagsHub registry.

Open the interactive API documentation at:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

## API Reference

### `GET /`

Returns a service description.

### `GET /health`

Reports whether the prediction and embedding models are loaded, along with the number of cached results.

### `POST /inference/analyze`

Request body:

```json
{
	"app_id": "com.supercell.clashofclans",
	"scraping_method": "new_reviews",
	"force_refresh": false
}
```

`scraping_method` accepts `new_reviews`, `relevant_reviews`, or `all_reviews`. `force_refresh` bypasses the process-local cache.

Example PowerShell request:

```powershell
Invoke-RestMethod -Method Post `
	-Uri http://localhost:8000/inference/analyze `
	-ContentType "application/json" `
	-Body '{"app_id":"com.supercell.clashofclans","scraping_method":"new_reviews"}'
```

The response includes `review_count`, `sentiment_percentages`, `top_reviews`, `topic_summary`, `llm_insight`, `insight_generated_at`, `execution_time_seconds`, and `cache_hit`.

### `DELETE /inference/cache/{app_id}`

Removes the cached result for an app. The optional `scraping_method` query parameter defaults to `new_reviews`.

```powershell
Invoke-RestMethod -Method Delete `
	-Uri "http://localhost:8000/inference/cache/com.supercell.clashofclans?scraping_method=new_reviews"
```

## Chrome Extension

1. Start the API locally or configure an accessible backend.
2. Open `chrome://extensions` in Chrome.
3. Enable **Developer mode**.
4. Select **Load unpacked** and choose the `extension` directory.
5. Open a Google Play app page, or paste its package ID into the extension.
6. Choose a review source and select **Analyze Reviews**.

The extension can detect the package ID from the active Play Store tab. Its current backend options are defined in [extension/popup.js](extension/popup.js), including the local service at `http://localhost:8000` and the configured cloud service. Update the cloud URL and [extension/manifest.json](extension/manifest.json) host permissions when deploying to a different backend.

## Data and ML Pipelines

### ETL from the labeled CSV

The standard ETL pipeline reads `datasource/raw/app_reviews_labeled.csv`, transforms the review text and labels, and appends deduplicated records to the `reviews_datalake` table:

```powershell
python -m src.pipelines.etl_pipeline
```

### Production-style ETL

The production pipeline scrapes reviews from selected apps, labels them with the configured transformer model, transforms them, and loads them into the database:

```powershell
python -m src.pipelines.etl_pro_pipeline
```

### Train and register a model

The ML pipeline fetches recent data, creates TF-IDF features, trains and evaluates a balanced `LinearSVC`, tracks the run with MLflow, and registers the model. A new model is promoted to the Champion alias only when it meets the configured promotion rules:

```powershell
python -m src.pipelines.ml_pipeline
```

Training settings, including the test split and MLflow model name, are in [config/settings.yml](config/settings.yml).

### Run inference directly

For command-line experimentation, the inference pipeline can be run without the API:

```powershell
python -m src.pipelines.inference_pipeline
```

This requires the same model registry, embedding model, and Groq configuration as the API.

## Docker

Build and run the production image:

```powershell
docker build -f dockerfile -t revuai:latest .
docker run --rm -p 8000:8000 --env-file .env revuai:latest
```

The container exposes port `8000`, runs one Uvicorn worker, and includes a health check against `/health`.

## Testing and Development

Run the test suite:

```powershell
python -m pytest
```

Useful development commands:

```powershell
make install-dev
make test
make test-coverage
make lint
make format
make clean
```

The test suite uses lightweight fakes for model loading, scraping, and LLM calls so API tests do not require network access or production credentials.

## Automation

RevuAI uses GitHub Actions for continuous integration, scheduled data refreshes, model retraining, and deployment. The workflows are defined in `.github/workflows/`.

### Continuous integration and deployment

On every push to `main`, the deployment workflow:

1. Checks out the repository and installs the Python 3.11 dependencies.
2. Runs the complete `pytest` suite.
3. Builds a production Docker image for `linux/amd64`.
4. Publishes both a commit-specific image and `latest` to GitHub Container Registry.
5. Connects to the configured EC2 host over SSH.
6. Pulls the new image, replaces the running container, and checks the `/health` endpoint.

The deployment workflow can also be started manually or called by the monthly training workflow. Deployment concurrency is serialized so two deployments cannot replace one another at the same time.

### Monthly data and retraining cycle

The production learning cycle runs on the first day of every month:

1. At `00:00 UTC`, `monthly_etl.yml` runs the production ETL pipeline. It scrapes reviews, labels them, transforms the records, and appends deduplicated data to the PostgreSQL/Supabase `reviews_datalake` table.
2. At `03:00 UTC`, `monthly_ml_training.yml` reads the latest data from the data lake and trains a new TF-IDF plus balanced `LinearSVC` sentiment model.
3. The training run is tracked in MLflow and the model is registered in DagsHub.
4. The new model is evaluated against the current Champion using the configured test F1 score.
5. The Champion alias is updated only when the new model satisfies the promotion rules in [config/settings.yml](config/settings.yml).
6. The workflow reads the pipeline's `CHAMPION_UPDATED=true|false` signal. If the Champion changed, it calls the deployment workflow; otherwise, the existing production service remains unchanged.

Both scheduled workflows also support manual execution with GitHub Actions' **Run workflow** control. Concurrency groups prevent overlapping ETL or training runs.

### Workflow summary

| Workflow | Trigger | Responsibility |
| --- | --- | --- |
| `ci.yml` | Push to `main` | Install dependencies and run tests |
| `deploy.yml` | Push to `main`, manual, or reusable call | Test, build, publish, and deploy the API |
| `monthly_etl.yml` | Monthly at 00:00 UTC or manual | Scrape, label, transform, and load reviews |
| `monthly_ml_training.yml` | Monthly at 03:00 UTC or manual | Train, evaluate, register, and conditionally deploy a model |

### Required repository secrets

Configure the repository secrets required by the workflows before enabling scheduled runs:

- `DATABASE_URL` for the production ETL workflow.
- `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` for MLflow tracking and model registration.
- `EC2_HOST`, `EC2_USERNAME`, and `EC2_SSH_KEY` for container deployment.

## Repository Layout

```text
main.py                     FastAPI application and startup lifecycle
router/                     API routes and request validation
src/inference/              Scraping, prediction, clustering, and LLM insight logic
src/etl/                    Extract, transform, and load components
src/ml/                     Training, MLflow tracking, and model registry logic
src/pipelines/              Runnable ETL, ML, and inference workflows
database/                   SQLAlchemy connection and review models
config/                     YAML training settings and LLM prompt constants
datasource/                 Source datasets and labeling utilities
extension/                  Chrome extension popup and manifest
tests/                      API, ETL, inference, and ML tests
```

## Operational Notes

- The inference cache is process-local and is cleared when the service restarts. It is not shared between multiple workers.
- Full inference can take several minutes because scraping, clustering, and LLM generation are performed in sequence.
- A missing `GROQ_API_KEY` prevents insight generation and causes full inference requests to fail.
- Google Play scraping depends on the availability and behavior of the upstream Play Store service.
- The extension's cloud endpoint is configured in source code. Use HTTPS and update host permissions before exposing a production deployment.
- Keep dependency versions and model registry credentials controlled in deployment environments for reproducible builds.

## License

See [LICENSE](LICENSE).
