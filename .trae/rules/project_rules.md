# Auto-sklearn Project Rules

## Development Commands

### Environment Setup
```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install development dependencies
make install-dev

# Install all dependencies (including docs, examples, benchmark)
make install-all
```

### Code Quality (Four-Layer Quality Gate)
```bash
# Layer 1: Code style check (ruff + black + isort)
make lint

# Layer 2: Static type check (mypy)
make type-check

# Layer 3: Security scan (bandit + vulture)
make security

# Layer 4: Unit tests
make test

# Run all quality checks
make check
```

### Formatting
```bash
# Format code
make format
```

### Testing
```bash
# Run unit tests
make test

# Run tests with coverage
make test-cov

# Run tests in parallel
make test-parallel

# Run benchmark tests
make test-benchmark
```

### Documentation
```bash
# Build documentation
make doc

# Serve documentation locally with hot reload
make doc-serve
```

### Build & Publish
```bash
# Build distribution
make build

# Publish to TestPyPI
make publish

# Publish to PyPI
make publish-prod
```

### Docker
```bash
# Build Docker image
make docker-build

# Run Docker container
make docker-run
```

### Cleanup
```bash
# Clean build artifacts
make clean

# Deep clean (including virtual environment)
make clean-all
```

## Pre-commit Hooks
Pre-commit hooks are automatically installed when running `make install-dev`.
To run manually:
```bash
make pre-commit
```

## VS Code Dev Container
1. Open project in VS Code
2. Click "Reopen in Container" when prompted
3. Wait for container to build and dependencies to install
4. Start coding!

## Project Structure
```
autosklearn/
├── .devcontainer/       # VS Code dev container config
├── .github/             # GitHub Actions workflows
├── autosklearn/         # Main package
│   ├── plugins/         # Plugin system
│   ├── pipeline/        # ML pipelines
│   └── ...
├── docs/                # MkDocs documentation
├── examples/            # Example scripts
├── scripts/             # Utility scripts
├── test/                # Test suite
│   └── benchmark/       # Performance benchmarks
├── Makefile             # Development tasks
├── mkdocs.yml           # Documentation config
├── pyproject.toml       # Poetry config + tool settings
└── requirements.txt     # Legacy requirements (deprecated)
```

## Key Files
- `pyproject.toml`: Poetry configuration, tool settings (ruff, mypy, pytest, etc.)
- `Makefile`: Development task automation
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `mkdocs.yml`: Documentation configuration
- `.devcontainer/`: VS Code dev container configuration
