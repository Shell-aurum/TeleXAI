# Use a lightweight Python base image
FROM python:3.14-slim
# Install the OpenMP library required by LightGBM/XGBoost
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the dashboard when the container starts
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]