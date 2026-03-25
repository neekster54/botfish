setup:
	uv sync
	pre-commit install

check:
	ruff format .
	ruff check .