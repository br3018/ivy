FROM python:3.14-slim-trixie

# Install UV from Astral SH's prebuilt image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync

# Presuming there is a `main.py` command provided by the project
CMD ["uv", "run", "main.py"]