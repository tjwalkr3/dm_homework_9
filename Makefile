VENV_DIR ?= .venv

install:
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install pandas

run:
	$(VENV_DIR)/bin/python weather.py

clean:
	rm -rf $(VENV_DIR)

reinstall: clean install

.PHONY: install run clean reinstall
