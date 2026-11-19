# exact patch tag (the check is also byte-identical on 3.12 / 3.14, but the shipped image is pinned)
FROM python:3.11.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# `docker build` needs network once (to install the pinned test dependency); `docker run --network none` is offline.
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY . .

# One official, fully offline verification command (full check, a few minutes). Mount a volume on /app/reports to keep
# the report: docker run --rm --network none -v "$PWD/reports:/app/reports" auditgym
# A closed run splits grader and agent into separate containers; see README "Trust boundary and deployment".
CMD ["python", "-m", "auditgym", "check"]
