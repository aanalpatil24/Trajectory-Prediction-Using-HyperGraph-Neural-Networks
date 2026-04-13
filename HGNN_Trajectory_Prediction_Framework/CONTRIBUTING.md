# Contributing to HGNN Trajectory Prediction

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/hgnn_trajectory_prediction.git
   cd hgnn_trajectory_prediction
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

## Code Style

We use:
- **Black** for code formatting
- **flake8** for linting

Format your code before committing:
```bash
make format
make lint
```

## Testing

Run tests before submitting:
```bash
make test
```

Add tests for new features in the `tests/` directory.

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Link to related issues
   - Test results
   - Example usage if applicable

## Code Structure

When adding new features:
- Place models in `src/models/`
- Place data utilities in `src/data/`
- Place training code in `src/training/`
- Place general utilities in `src/utils/`
- Add tests in `tests/`
- Update documentation in `README.md`

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- OS information
- Minimal code to reproduce the issue
- Error messages and stack traces

## Feature Requests

We welcome feature requests! Please:
- Describe the use case
- Explain why current solutions are insufficient
- Provide examples if possible

## Questions?

Feel free to open an issue for questions or join discussions.
