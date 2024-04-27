# Use Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies if required
RUN pip install -r requirements.txt

# Execute the run.py script
CMD ["python", "run.py"]
