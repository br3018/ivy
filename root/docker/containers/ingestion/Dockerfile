FROM python:3.13-slim-trixie

# Install system dependencies (poppler-utils for pdf2image and git)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV from Astral SH's prebuilt image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Set UV environment variables
ENV UV_LINK_MODE=copy

# Copy the project into the image
COPY . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Presuming there is a `main.py` command provided by the project
CMD ["uv", "run", "main.py"]