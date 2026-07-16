.PHONY: install dev test lint format demo docker-up docker-down
install:
	python -m pip install -e ".[dev]"
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
test:
	python -m pytest -q
lint:
	python -m ruff check app tests
format:
	python -m ruff format app tests
demo:
	python scripts/generate_sample_video.py && python scripts/run_demo.py
docker-up:
	docker compose up --build
docker-down:
	docker compose down
