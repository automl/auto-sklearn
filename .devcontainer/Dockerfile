ARG VARIANT="3.10"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    curl \
    wget \
    git \
    git-lfs \
    vim \
    nano \
    htop \
    tree \
    jq \
    ripgrep \
    fd-find \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV BLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.5.1
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry config installer.max-workers 4

WORKDIR /workspace

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --with dev,docs,examples,benchmark || true

RUN echo 'export PATH="/workspace/.venv/bin:$PATH"' >> ~/.bashrc && \
    echo 'export PATH="/workspace/.venv/bin:$PATH"' >> ~/.zshrc

EXPOSE 8888 8000

CMD ["sleep", "infinity"]
