# Contributing to ContextTape

Thank you for considering contributing to ContextTape! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to info@nuterralabs.com.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, package version)
- **Code samples** or test cases demonstrating the issue

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description** of the proposed feature
- **Use cases** and examples of how it would be used
- **Potential implementation approaches** if you have ideas
- **Why this enhancement would be useful** to most users

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**, following the coding standards below
3. **Add tests** for any new functionality
4. **Update documentation** to reflect your changes
5. **Ensure tests pass** by running `pytest`
6. **Submit your pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/contexttape.git
cd contexttape

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Run tests
pytest tests/ -v --cov=contexttape

# Run linting
ruff check src/
black --check src/
```

## Coding Standards

### Python Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function arguments and return values
- Maximum line length: **100 characters**
- Use **black** for code formatting
- Use **ruff** for linting

### Documentation

- Add **docstrings** to all public functions, classes, and modules
- Use **Google-style docstrings**
- Include **examples** in docstrings when appropriate
- Update **README.md** for user-facing changes

### Testing

- Write **unit tests** for new functionality
- Aim for **>80% code coverage**
- Use **pytest** fixtures for test setup
- Include **integration tests** for complex features
- Test **edge cases** and error conditions

### Commits

- Use **clear, descriptive commit messages**
- Follow the format: `type(scope): description`
  - Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`
  - Example: `feat(storage): add batch append operation`
- Keep commits **focused and atomic**
- Reference issues: `fix(search): correct cosine similarity (#123)`

## Project Structure

```
contexttape/
├── src/
│   └── contexttape/
│       ├── __init__.py      # Package exports
│       ├── storage.py       # Core storage logic
│       ├── embed.py         # Embedding utilities
│       ├── search.py        # Search and retrieval
│       ├── cli.py           # Command-line interface
│       ├── benchmark.py     # Performance benchmarking
│       ├── integrations.py  # Framework integrations
│       └── ...
├── tests/
│   ├── test_storage.py      # Storage tests
│   ├── test_search.py       # Search tests
│   └── ...
├── examples/
│   ├── quickstart.py
│   ├── advanced_usage.py
│   └── ...
├── docs/
│   └── ...
└── pyproject.toml
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=contexttape --cov-report=html

# Run specific test file
pytest tests/test_storage.py

# Run specific test
pytest tests/test_storage.py::TestISStore::test_append_text

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x
```

### Writing Tests

```python
import pytest
from contexttape import ISStore

def test_feature_name():
    """Test description."""
    # Arrange
    store = ISStore("/tmp/test")
    
    # Act
    result = store.some_method()
    
    # Assert
    assert result == expected_value
    
    # Cleanup (if needed)
    # Use pytest fixtures instead when possible
```

## Performance Considerations

- **Benchmark** performance-critical changes
- **Profile** code to identify bottlenecks
- Consider **memory usage** for large datasets
- Test with **realistic data sizes**
- Document **performance characteristics**

## Documentation

### Building Documentation

```bash
cd docs/
pip install -r requirements-docs.txt
mkdocs serve
```

### Documentation Guidelines

- Keep documentation **up to date** with code
- Include **code examples** that actually work
- Add **diagrams** for complex concepts
- Write for **different skill levels**
- Cross-reference related sections

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.x.x`
4. Push tag: `git push origin v0.x.x`
5. GitHub Actions will build and publish to PyPI

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community discussion
- **Email**: info@nuterralabs.com for private inquiries

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Attribution

Contributors will be acknowledged in:
- `CHANGELOG.md` for their contributions
- GitHub's contributor graph
- Release notes

Thank you for contributing to ContextTape! 🎉
