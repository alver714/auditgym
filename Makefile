PY ?= .venv/bin/python

.PHONY: check quick test generate demo docker

check:
	$(PY) -m auditgym check

quick:
	$(PY) -m auditgym check --quick

test:
	$(PY) -m pytest -q

generate:
	$(PY) -m auditgym generate

demo:
	$(PY) -m auditgym demo

docker:
	docker build -t auditgym .
	mkdir -p reports
	docker run --rm --network none -v "$(CURDIR)/reports:/app/reports" auditgym
