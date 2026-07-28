.PHONY: validate render check test all

validate:
	python3 scripts/validate_catalog.py

render: validate
	python3 scripts/render_catalog.py

test:
	python3 -m unittest discover -s scripts -p "test_*.py"

check: validate test
	python3 scripts/render_catalog.py --check

all: render
