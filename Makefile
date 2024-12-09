# Define variables
APP_NAME = app.py
PORT = 3000
VENV = venv
PACKAGE_MANAGER = pip

# Default target
.PHONY: all
all: install run

# Install dependencies
.PHONY: install
install:
# 	Create a virtual environment if it doesn't exist
	if [ ! -d $(VENV) ]; then \
		python3 -m venv $(VENV); \
	fi
# 	Activate the virtual environment and install requirements
	. $(VENV)/bin/activate && $(PACKAGE_MANAGER) install -r requirements.txt

# Run the application
.PHONY: run
run:
# 	Activate the virtual environment and run the Flask app
	. $(VENV)/bin/activate && FLASK_APP=$(APP_NAME) flask run --host=0.0.0.0 --port=$(PORT)