# Makefile for Flask Site (Python venv)

# Python interpreter and virtual environment directory
PYTHON = python
VENV_DIR = venv

# Requirements file
REQUIREMENTS = requirements.txt

# Create virtual environment
create-venv:
	$(PYTHON) -m venv $(VENV_DIR)

# Install dependencies in virtual environment
install-requirements: create-venv
	$(VENV_DIR)\Scripts\pip install -r $(REQUIREMENTS)

# Activate the virtual environment
activate-venv:
	$(VENV_DIR)\Scripts\activate

# Run the Flask development server (assumes app.py or main.py is the entry point)
run-flask:
	$(VENV_DIR)\Scripts\python app.py

# Clean up virtual environment (optional)
clean:
	rm -rf $(VENV_DIR)

# Lint your code (optional, install flake8 if necessary)
lint:
	$(VENV_DIR)\Scripts\python -m flake8 .

# Test your Flask app (optional, example using pytest)
test:
	$(VENV_DIR)\Scripts\python -m pytest
