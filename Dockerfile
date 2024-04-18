# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose any necessary ports
# EXPOSE <port>

# Define environment variable
# ENV NAME value

# Run script to start your application
CMD ["python", "era5_swvl1_sr.py"]
