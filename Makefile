.PHONY: run run-root snapshot install bundle clean test

PYTHON ?= python3

run:
	$(PYTHON) -m zenmon

run-root:
	sudo $(PYTHON) -m zenmon

snapshot:
	$(PYTHON) -m zenmon --snapshot

install:
	pip install .

install-dev:
	pip install -e .

bundle:
	$(PYTHON) -c 'import re; \
	header = "#!/usr/bin/env python3\n\"\"\"ZenMon: Serene telemetry TUI\nLicense: MIT\n\"\"\"\n"; \
	ai = open("zenmon/ai_inspector.py").read(); \
	hal = open("zenmon/hal.py").read().replace("from .ai_inspector import detect_ai_tag", "# [Inlined ai]"); \
	app = open("zenmon/app.py").read(); \
	app = re.sub(r"from \.hal import \([\s\S]*?\)", "# [Inlined hal]", app); \
	app = re.sub(r"from \.hal import [^\n]+", "# [Inlined hal]", app); \
	app = app.replace("from .ai_inspector import analyze_ai_summary", "# [Inlined ai]"); \
	main = open("zenmon/__main__.py").read(); \
	main = re.sub(r"from \.app import [^\n]+", "# [Inlined app]", main); \
	main = main.replace("from . import __version__", "__version__ = \"2.0.0\""); \
	out = f"{header}\n# SECTION 1\n{ai}\n# SECTION 2\n{hal}\n# SECTION 3\n{app}\n# SECTION 4\n{main}"; \
	open("standalone/zenmon.py", "w").write(out); \
	print("Standalone bundled successfully!")'
	chmod +x standalone/zenmon.py

test:
	$(PYTHON) -m py_compile zenmon/*.py standalone/zenmon.py
	$(PYTHON) standalone/zenmon.py --version
	$(PYTHON) -m zenmon --version

clean:
	rm -rf build dist *.egg-info __pycache__ zenmon/__pycache__
