.DEFAULT_GOAL := help
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Formatters

.PHONY: format-black
format-black: ## run black (code formatter)
	@black .

.PHONY: format-isort
format-isort: ## run isort (imports formatter)
	@isort .

.PHONY: format
format: format-black format-isort ## run all formatters

##@ Linters

.PHONY: lint-black
lint_black: ## run black in linting mode
	@black . --check

.PHONY: lint-isort
lint-isort: ## run isort in linting mode
	@isort . --check

.PHONY: lint-flake8
lint-flake8: ## run flake8 (code linter)
	@flake8 ./object_detection_task

.PHONY: lint-mypy
lint-mypy: ## run mypy (static-type checker)
	@mypy --config-file pyproject.toml ./object_detection_task

.PHONY: lint-mypy-report
lint-mypy-report: ## run mypy & create report
	@mypy --config-file pyproject.toml . --html-report ./mypy_html

lint: lint-black lint-isort lint-flake8 lint-mypy

##@ Testing

.PHONY: unit-tests
unit-tests: ## run unit-tests
	@pytest

.PHONY: unit-tests-cov
unit-tests-cov: ## run unit-tests with coverage
	@pytest --cov=object_detection_task --cov-report term-missing --cov-report=html

.PHONY: unit-tests-cov-fail
unit-tests-cov-fail: ## run unit-tests with coverage and cov-fail level
	@pytest --cov=object_detection_task --cov-report term-missing --cov-report=html --cov-fail-under=80 --junitxml=pytest.xml | tee pytest-coverage.txt
##@ Clean-up

clean-cov: ## run cleaning from reports
	@rm -rf .coverage
	@rm -rf htmlcov
	@rm -rf pytest.xml
	@rm -rf pytest-coverage.txt
	@rm -rf coverage.xml
	@rm -rf mypy_html

clean-docs: ## remove output files from mkdocs
	@rm -rf site
