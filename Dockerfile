FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
# Pass your key at run time:  docker run -p 7860:7860 -e GEMINI_API_KEY=... insurance-advisor-rag
CMD ["python", "app.py"]
