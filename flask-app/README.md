# Classification Application Setup

This README outlines the steps necessary to get this font classification application up and running.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3
- pip (Python package installer)

## Virtual Environment Setup

### For Linux or MacOS

1. Navigate to your project directory: `cd path/to/your/project`
2. Create the virtual environment: `python3 -m venv env`
3. Activate the virtual environment: `source env/bin/activate`

### For Windows

1. Navigate to your project directory: `cd path\to\your\project`
2. Create the virtual environment: `py -3 -m venv env`
3. Activate the virtual environment: `.\env\Scripts\activate`

## Install Required Packages

With the virtual environment activated, install the required packages using:

`pip install -r requirements.txt`

This command reads the requirements.txt file in the project directory and installs all the necessary packages.

## Running the Flask Application

To run the Flask application, go to the app directory and use the following command: `flask run`

This will start a development server and you should see output similar to this:

`* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)`

You can now access the application in your web browser at http://127.0.0.1:5000/.
