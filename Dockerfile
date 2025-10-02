# Use Python 3.11 as base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Set environment variable
ENV PORT=8080

# Run the application
CMD ["python", "-c", "import streamlit; streamlit.run('examples/app_streamlit.py', server_port=8080, server_address='0.0.0.0', server_headless=True, server_enableCORS=False, server_enableXsrfProtection=False)"]