# CLAUDE.md - Project Instructions for Claude

> Claude reads this file automatically at the start of every conversation.

## Project Overview

This is an **end-to-end ML classification project** for learning purposes.
The goal is to learn ML deployment: from model training to production on AWS/K8s.

## Current Progress

| Step | Description | Status |
|------|-------------|--------|
| 0 | Project Setup (git, config, logging, pre-commit) | ✅ Done |
| 1 | EDA - Exploratory Data Analysis | ✅ Done |
| 2 | ML Model Training | ✅ Done |
| 3 | FastAPI Backend | ✅ Done |
| 4 | Streamlit Frontend | ✅ Done |
| 5 | Docker | ⏳ Next |
| 6 | Kubernetes | ⬜ Pending |
| 7 | AWS (ECR, EKS) | ⬜ Pending |
| 8 | CI/CD (GitHub Actions) | ⬜ Pending |
| 9 | Monitoring (Prometheus, Grafana) | ⬜ Pending |

## User Preferences

1. **Learning Focus**: This is a learning project. Always:
   - Add detailed comments explaining concepts
   - Update STUDY.md with explanations
   - Show "why" not just "how"

2. **Professional Practices**: Follow industry standards:
   - Use src/config.py for configuration (not hardcoded values)
   - Use src/logger.py for logging (not print statements)
   - Write tests for new features

3. **Step-by-Step**: Work incrementally:
   - Complete one step at a time
   - Wait for user approval before next step
   - Update STUDY.md after each step

## Key Files

| File | Purpose |
|------|---------|
| `STUDY.md` | Detailed learning documentation |
| `README.md` | Project overview |
| `src/config.py` | Configuration management |
| `src/logger.py` | Logging setup |
| `src/ml/train.py` | Model training |
| `src/ml/predict.py` | Prediction module |
| `src/api/main.py` | FastAPI backend |
| `src/frontend/app.py` | Streamlit frontend |
| `Makefile` | Common commands |

## Common Commands

```bash
make help       # Show all commands
make train      # Train ML model
make api        # Start FastAPI (port 8000)
make frontend   # Start Streamlit (port 8501)
make test       # Run tests
make lint       # Check code quality
make format     # Format code
```

## Tech Stack

- **ML**: Python, scikit-learn, pandas, numpy
- **API**: FastAPI, Pydantic, uvicorn
- **Frontend**: Streamlit
- **Containers**: Docker, docker-compose
- **Orchestration**: Kubernetes
- **Cloud**: AWS (ECR, EKS)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana

## Resume Instructions

If starting a new conversation, say:
> "Continue ML project from Step X"

Claude will read this file and STUDY.md to understand the context.
