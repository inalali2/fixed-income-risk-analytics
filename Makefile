.PHONY: install test lint sample live dashboard data

install:
	pip install -e ".[dev,dashboard]"

test:
	pytest -q

lint:
	ruff check src tests dashboard scripts

sample:
	fi-risk report --source sample --out reports/sample

data:
	fi-risk update --sources boc credit

live:
	fi-risk report --source boc --out reports/latest

dashboard:
	streamlit run dashboard/app.py
