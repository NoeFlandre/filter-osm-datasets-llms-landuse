# CUDA runtime image with the locked GPU environment (CI smoke + local reproduction).
# Grid'5000 nodes use the same lockfile through scripts/node_job.sh (ADR-0008).
FROM nvidia/cuda:13.4.2-cudnn-runtime-ubuntu24.04@sha256:2aa09bb7626c4bff7d0a08b0028edb8dc254d0252a8b67cd386966a864fafac2
COPY --from=ghcr.io/astral-sh/uv:0.11@sha256:77280f2f771df71f90786c314fe1bbc1e023feac652969bbf139c280babf2eb7 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra gpu --python 3.12
ENV PATH=/opt/venv/bin:$PATH
ENTRYPOINT ["luf"]
CMD ["--help"]
