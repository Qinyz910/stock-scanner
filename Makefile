.PHONY: format lint typecheck test coverage frontend-lint frontend-typecheck frontend-test

lint:
	ruff check .
	black --check .

format:
	black .

typecheck:
	mypy .

test:
	pytest

coverage:
	pytest --cov

frontend-lint:
	npm run lint --prefix frontend

frontend-typecheck:
	npm run typecheck --prefix frontend

frontend-test:
	npm run test --prefix frontend
