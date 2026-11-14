#!/bin/bash
set -e

# 1. Start PostgreSQL in Docker
docker-compose up -d

# 2. Activate your Python virtual environment
source venv/bin/activate

# 3. Start FastAPI in the background
uvicorn main:app --host localhost --port 8000 &
API_PID=$!

# Ensure FastAPI stops when you press Ctrl+C
trap "kill $API_PID" EXIT

# 4. Start Streamlit UI in the foreground
streamlit run app.py --server.port 8501
