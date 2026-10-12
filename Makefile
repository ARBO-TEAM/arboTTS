PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
PORT ?= 8000
HOST ?= 0.0.0.0

.PHONY: help install install-web install-all test serve demo demo-all docker-build docker-run clean

help:
	@printf '%s\n' \
	  'ArboTTS v2.0 - Developer & Operations Commands' \
	  '' \
	  'Usage:' \
	  '  make install       Install core package in editable mode' \
	  '  make install-web   Install package with Web UI & REST API dependencies' \
	  '  make install-all   Install all dependencies (web + training)' \
	  '  make serve         Run Web UI Studio and REST API server on port $(PORT)' \
	  '  make test          Run test suite with pytest' \
	  '  make demo          Run quick speech synthesis sample' \
	  '  make demo-all      Generate audition samples for all 7 personas' \
	  '  make docker-build  Build lightweight standalone CPU Docker image' \
	  '  make docker-run    Run Docker container on http://localhost:$(PORT)' \
	  '  make clean         Clean build artifacts and caches'

install:
	$(PYTHON) -m pip install -e .

install-web:
	$(PYTHON) -m pip install -e ".[web]"

install-all:
	$(PYTHON) -m pip install -e ".[web,train]"

serve:
	PYTHONPATH=. $(PYTHON) -m arbotts.server --host $(HOST) --port $(PORT)

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests -v

demo:
	PYTHONPATH=. $(PYTHON) -m arbotts.cli "Halo, selamat datang di ArboTTS Studio!" -o demo_arbo.wav -sp arbo -s 0.85
	@echo "Audio generated at demo_arbo.wav"

demo-all:
	@mkdir -p demo_outputs
	PYTHONPATH=. $(PYTHON) -m arbotts.cli "Halo, ini Arbo, penyiar berita resmi Anda." -o demo_outputs/arbo.wav -sp arbo -s 0.85
	PYTHONPATH=. $(PYTHON) -m arbotts.cli "Halo, saya Bimo, narator baritone berwibawa." -o demo_outputs/bimo.wav -sp bimo -s 0.85
	PYTHONPATH=. $(PYTHON) -m arbotts.cli "Hai! Saya Dimas, asisten kasual dan energik." -o demo_outputs/dimas.wav -sp dimas -s 0.90
	PYTHONPATH=. $(PYTHON) -m arbotts.cli "Selamat malam, saya Danu, narator santun dan lembut." -o demo_outputs/danu.wav -sp danu -s 0.85
	PYTHONPATH=. $(PYTHON) -m arbotts.cli "Selamat pagi, saya Arba, presenter berita profesional." -o demo_outputs/arba.wav -sp arba -s 0.85
	PYTHONPATH=. $(PYTHON) -m arbotts.cli "Hai! Saya Sita, asisten cerdas yang siap membantu." -o demo_outputs/sita.wav -sp sita -s 0.90
	PYTHONPATH=. $(PYTHON) -m arbotts.cli "Selamat malam, saya Maya, mari kita dengarkan kisah ini." -o demo_outputs/maya.wav -sp maya -s 0.80
	@echo "All 7 persona samples generated in demo_outputs/"

docker-build:
	docker build -t arbotts:v2.0 -t arbotts:latest .

docker-run:
	docker run --rm -p $(PORT):8000 arbotts:latest

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
