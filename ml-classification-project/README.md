# Iris Classification - End-to-End ML Project

A production-ready machine learning classification project demonstrating the complete ML lifecycle from development to deployment.

## Overview

This project classifies iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal measurements. It serves as a learning resource for building end-to-end ML systems.

---

## Project Steps & When To Do What

> **IMPORTANT**: Professional setup (Step 0) should be done FIRST, before writing any code!

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROJECT IMPLEMENTATION ORDER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 0: PROJECT SETUP (DO THIS FIRST!)                                    │
│  ──────────────────────────────────────                                    │
│  Before writing ANY code, set up:                                          │
│                                                                             │
│  □ Git repository          → git init                                      │
│  □ .gitignore              → Prevent committing secrets/junk               │
│  □ Virtual environment     → python -m venv venv                           │
│  □ requirements.txt        → Pin dependencies                              │
│  □ .env.example            → Document required env vars                    │
│  □ src/config.py           → Centralized configuration                     │
│  □ src/logger.py           → Proper logging (not print!)                   │
│  □ pyproject.toml          → Tool configurations                           │
│  □ .pre-commit-config.yaml → Code quality hooks                            │
│  □ Makefile                → Common commands                               │
│  □ README.md               → Project documentation                         │
│  □ STUDY.md                → Learning notes (optional)                     │
│                                                                             │
│  WHY FIRST?                                                                │
│  - Git tracks all changes from day 1                                       │
│  - Config/logging used by ALL other code                                   │
│  - Pre-commit catches issues immediately                                   │
│  - Team members can onboard quickly                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: Data Exploration (EDA)                                            │
│  □ Explore dataset                                                         │
│  □ Document findings                                                       │
│                                                                             │
│  STEP 2: ML Model Development                                              │
│  □ Train models                                                            │
│  □ Evaluate & save best model                                              │
│                                                                             │
│  STEP 3: API Development (FastAPI)                                         │
│  □ Create prediction endpoints                                             │
│  □ Add tests                                                               │
│                                                                             │
│  STEP 4: Frontend (Streamlit)                                              │
│  □ Build user interface                                                    │
│                                                                             │
│  STEP 5: Containerization (Docker)                                         │
│  □ Dockerize all services                                                  │
│                                                                             │
│  STEP 6: Orchestration (Kubernetes)                                        │
│  □ Write K8s manifests                                                     │
│                                                                             │
│  STEP 7: Cloud Infrastructure (AWS)                                        │
│  □ Set up ECR, EKS                                                         │
│                                                                             │
│  STEP 8: CI/CD (GitHub Actions)                                            │
│  □ Automate testing & deployment                                           │
│                                                                             │
│  STEP 9: Monitoring                                                        │
│  □ Add Prometheus metrics                                                  │
│  □ Set up Grafana dashboards                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### When to Add What (Summary Table)

| Item | When to Add | Why |
|------|-------------|-----|
| **Git + .gitignore** | **FIRST** | Track all changes, never commit secrets |
| **Virtual env + requirements.txt** | **FIRST** | Reproducible environment |
| **.env.example + config.py** | **FIRST** | All code uses config from start |
| **logger.py** | **FIRST** | All code uses logging from start |
| **pyproject.toml** | **FIRST** | Tool configs ready for use |
| **.pre-commit-config.yaml** | **FIRST** | Catch issues on every commit |
| **Makefile** | **FIRST** | Standard commands from day 1 |
| **README.md** | **FIRST** (update as you go) | Documentation |
| tests/ | With each feature | Test as you build |
| Dockerfile | Step 5 | After code works locally |
| k8s/ manifests | Step 6 | After Docker works |
| .github/workflows/ | Step 8 | After K8s works locally |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Cloud | AWS (EKS, ECR) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |

---

## Project Structure

```
ml-classification-project/
│
├── .gitignore               # ← Step 0: Git ignore rules
├── .env.example             # ← Step 0: Env vars template
├── .env                     # ← Step 0: Local env vars (not committed!)
├── .pre-commit-config.yaml  # ← Step 0: Pre-commit hooks
├── pyproject.toml           # ← Step 0: Tool configurations
├── Makefile                 # ← Step 0: Common commands
├── README.md                # ← Step 0: Documentation
├── STUDY.md                 # ← Step 0: Learning guide
├── requirements.txt         # ← Step 0: Production dependencies
├── requirements-dev.txt     # ← Step 0: Dev dependencies
│
├── src/
│   ├── config.py            # ← Step 0: Configuration management
│   ├── logger.py            # ← Step 0: Logging setup
│   │
│   ├── ml/                  # ← Steps 1-2: Machine learning
│   │   ├── eda.py           #    Step 1: Data exploration
│   │   ├── train.py         #    Step 2: Model training
│   │   └── predict.py       #    Step 2: Prediction module
│   │
│   ├── api/                 # ← Step 3: FastAPI backend
│   │   └── main.py
│   │
│   └── frontend/            # ← Step 4: Streamlit app
│       └── app.py
│
├── models/                  # ← Step 2: Saved models
├── data/                    # ← Step 1: Datasets
├── tests/                   # ← Steps 1-4: Tests (add with each feature)
│
├── docker/                  # ← Step 5: Dockerfiles
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── k8s/                     # ← Step 6: Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
│
└── .github/workflows/       # ← Step 8: CI/CD pipelines
    └── ci.yml
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip
- Git

### Step 0: Initial Setup (Do This First!)

```bash
# Clone the repository
git clone <repository-url>
cd ml-classification-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env

# Verify setup
make help
```

### Running the Application

```bash
# 1. Train the model (Step 2)
make train

# 2. Start the API (Step 3)
make api
# API docs: http://localhost:8000/docs

# 3. Start the frontend (Step 4)
make frontend
# UI: http://localhost:8501
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

---

## Development Workflow

### Daily Development

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Write code...

# 3. Format & lint
make format
make lint

# 4. Run tests
make test

# 5. Commit (pre-commit hooks run automatically)
git add .
git commit -m "Add feature"

# 6. Push & create PR
git push origin feature/my-feature
```

### Available Make Commands

```bash
make help          # Show all commands
make install       # Install production deps
make install-dev   # Install dev deps + pre-commit
make train         # Train ML model
make api           # Start FastAPI server
make frontend      # Start Streamlit
make test          # Run tests
make test-cov      # Tests with coverage
make lint          # Run all linters
make format        # Auto-format code
make security      # Security scan
make docker-build  # Build containers
make clean         # Remove generated files
```

---

## Learning Guide

See [STUDY.md](STUDY.md) for detailed explanations of:
- Each step's implementation
- Key concepts and theory
- Code explanations
- Best practices

---

## License

MIT

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run `make lint` and `make test`
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request
