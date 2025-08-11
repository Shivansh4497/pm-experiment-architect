# Use an official Python runtime as a parent image
# We use a specific version to ensure consistency
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by reportlab
# This step has root access, so it will succeed.
RUN apt-get update && apt-get install -y \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    gcc \
    python3-dev

# Copy the requirements file and install the Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "main.py"]
