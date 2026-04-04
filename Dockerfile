FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir flask scikit-learn pandas numpy pytest tensorflow==2.20.0

COPY . .

RUN python model/rebuild_model.py

EXPOSE 8080

CMD ["python", "app.py"]
