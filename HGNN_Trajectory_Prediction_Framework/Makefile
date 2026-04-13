# Makefile for HGNN Trajectory Prediction

.PHONY: help install test clean train evaluate demo format lint

help:
	@echo "HGNN Trajectory Prediction - Available Commands:"
	@echo ""
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run all tests"
	@echo "  make demo        - Run quick demo"
	@echo "  make train       - Train model (quick)"
	@echo "  make evaluate    - Evaluate model"
	@echo "  make format      - Format code with black"
	@echo "  make lint        - Lint code with flake8"
	@echo "  make clean       - Clean generated files"
	@echo ""

install:
	pip install -r requirements.txt
	@echo "Installation complete!"

test:
	python -m pytest tests/ -v

demo:
	python scripts/demo.py

train:
	python scripts/train.py --dataset synthetic --epochs 10 --batch_size 16

evaluate:
	python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset synthetic --visualize

format:
	black src/ scripts/ tests/

lint:
	flake8 src/ scripts/ tests/ --max-line-length=100

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf outputs/logs/*
	rm -rf checkpoints/*.pt
	@echo "Cleaned up generated files"
