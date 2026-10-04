FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag.py api.py ./
# Ollama runs on the host. Linux hosts: add `--add-host=host.docker.internal:host-gateway` to docker run.
ENV OLLAMA_URL=http://host.docker.internal:11434
VOLUME ["/app/index", "/app/data"]
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
