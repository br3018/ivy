FROM python:3.14-slim-trixie

# Install ca-certificates and update
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

# Copy Zscaler root certificate if it exists (for corporate networks)
COPY zscaler-root-ca.cer* /tmp/zscaler-root-ca.cer
RUN if [ -s /tmp/zscaler-root-ca.cer ]; then \
        cp /tmp/zscaler-root-ca.cer /usr/local/share/ca-certificates/zscaler-root-ca.crt && \
        update-ca-certificates && \
        echo "Zscaler certificate installed successfully"; \
    else \
        echo "No Zscaler certificate found or file is empty - skipping (this is OK if not behind Zscaler)"; \
    fi

# Set environment variables for Python/pip to use system certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Install UV from Astral SH's prebuilt image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

