# Slim rather than alpine: scikit-learn needs glibc, and musl wheels either do
# not exist or must be compiled, which blows past Render's free build limits.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Requirements first so the dependency layer is cached across code changes —
# scikit-learn is the slow install and should not be repeated on every push.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts

# Non-root: the container has no reason to write anywhere, and Render's disk is
# ephemeral regardless.
RUN useradd --create-home --uid 10001 cascade && chown -R cascade:cascade /app
USER cascade

EXPOSE 8000

# Single worker on purpose. The free tier's 512 MB cannot hold two copies of the
# model plus the interpreter, and the workload is I/O-bound on provider calls,
# which one async worker handles well.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
