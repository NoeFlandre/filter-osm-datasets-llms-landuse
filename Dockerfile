# CUDA runtime image with the locked GPU environment (CI smoke + local reproduction).
# Grid'5000 nodes use the same lockfile through scripts/node_job.sh (ADR-0008).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra gpu --python 3.12
ENV PATH=/opt/venv/bin:$PATH
ENTRYPOINT ["luf"]
CMD ["--help"]
