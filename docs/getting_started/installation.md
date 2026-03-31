# Installation

## Prerequisites

auto-sklearn requires:

- Python (>= 3.7)
- Linux operating system (compatibility with macOS and Windows is limited)
- GCC (for compiling some dependencies)

## Installing auto-sklearn

### Using pip

```bash
pip install auto-sklearn
```

### Using Poetry

```bash
poetry add auto-sklearn
```

### From Source

```bash
git clone https://github.com/automl/auto-sklearn.git
cd auto-sklearn
pip install -e .
```

## Development Installation

For development, we recommend using the Poetry package manager:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/automl/auto-sklearn.git
cd auto-sklearn

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

## Docker Installation

We provide a Docker image for easy deployment:

```bash
# Pull the image
docker pull automl/auto-sklearn:latest

# Run a container
docker run -it --rm automl/auto-sklearn:latest python -c "import autosklearn; print(autosklearn.__version__)"
```

## Verifying Installation

To verify that auto-sklearn is installed correctly, run:

```python
import autosklearn
print(f"auto-sklearn version: {autosklearn.__version__}")
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: If you encounter errors about missing libraries, try installing the system dependencies first:

   ```bash
   sudo apt-get install build-essential gcc g++ gfortran libopenblas-dev liblapack-dev
   ```

2. **Python version**: auto-sklearn requires Python 3.7 or higher. Check your Python version:

   ```bash
   python --version
   ```

3. **Operating system**: auto-sklearn is primarily developed and tested on Linux. macOS and Windows compatibility is limited.

For more troubleshooting, please check the [GitHub issues](https://github.com/automl/auto-sklearn/issues) or create a new one.
