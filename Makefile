NAME = cmm

install:
	uv sync
	uv lock

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache
	rm -rf data/output

lint:
	uv run python -m flake8 .
	uv run python -m mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run python -m flake8 .
	uv run python -m mypy . --strict

test:
	uv run python -m pytest -q

.PHONY: install run debug clean lint lint-strict test
