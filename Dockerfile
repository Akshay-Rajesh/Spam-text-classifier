# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
# Copy only the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Create a non-root user and switch to it
RUN useradd -m myuser
USER myuser

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "predict.py"]