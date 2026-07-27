UV_RUN := uv run

.PHONY: format check dev-check

format:
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .

check:
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .
	$(UV_RUN) mypy
	$(UV_RUN) pytest
	git diff --check

dev-check: format
	$(MAKE) check