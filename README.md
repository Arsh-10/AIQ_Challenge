# AIQ_Challenge

# Image Processing with FastAPI

This project provides a simple image processing API built with FastAPI, allowing users to upload CSV files containing image data, process the images, and store them in a SQLite database. The API also provides endpoints for retrieving frames based on depth range and applying color maps to the images.

## Python Files

The project consists of the following Python files:

- `main.py`: Contains two main FastAPI application code in two different folders as per challenge.
- `requirements.txt`: Lists the Python dependencies required by the application.

## Docker

The project has been containerized using Docker for easy deployment. To run the application using Docker, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repository.git

2. Navigate to the project directory:
  cd your-repository

3. Build the Docker image:

  docker build -t image-processing-api .

4. Run the Docker container:
   docker run -d -p 8000:8000 image-processing-api

