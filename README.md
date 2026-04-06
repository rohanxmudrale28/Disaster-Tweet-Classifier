# 🚨 Disaster Tweet Classifier

A deep learning web application that classifies tweets as **disaster** or **not disaster** using a Bidirectional LSTM model built with TensorFlow and served via Flask.

---

## 👥 Team

| Name  | Role |
|-------|------|
| Rohan | Project Lead — Model, Flask API, Docker, Jenkins, Kubernetes |
| Shawn | Documentation & README |
| Parth | Unit Testing |

---

## 🧠 Model Architecture

- **Embedding Layer** → converts words to 100-dim vectors
- **Bidirectional LSTM (64 units)** → captures context in both directions
- **Dropout (0.3)** → prevents overfitting
- **Bidirectional LSTM (32 units)**
- **Dense (64, ReLU)** + **Dropout (0.3)**
- **Output (Sigmoid)** → binary classification

Trained on the [Kaggle NLP Disaster Tweets dataset](https://www.kaggle.com/competitions/nlp-getting-started).

---

## 📁 Project Structure

```
disaster-tweet-classifier/
├── app.py                  # Flask REST API + browser UI
├── model/
│   └── train.py            # Training script (saves model + tokenizer)
├── data/
│   ├── train.csv           # Training data
│   └── test.csv            # Test data
├── tests/
│   └── test_app.py         # Unit tests (Pytest)
├── k8s/
│   ├── deployment.yaml     # Kubernetes Deployment (2 replicas)
│   └── service.yaml        # Kubernetes Service (NodePort)
├── Dockerfile              # Docker image definition
├── Jenkinsfile             # CI/CD pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/disaster-tweet-classifier.git
cd disaster-tweet-classifier

# Install dependencies
pip install -r requirements.txt

# Place train.csv and test.csv inside data/
# Then train the model
python model/train.py

# Start the Flask server
python app.py
```

Visit `http://localhost:5000` in your browser.

---

### 2. Run with Docker

```bash
# Build image
docker build -t disaster-tweet-classifier .

# Run container
docker run -p 5000:5000 disaster-tweet-classifier

# Test the API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tweet": "There is a wildfire near my house!"}'
```

---

### 3. Jenkins CI/CD

Every `git push` to `main` triggers the Jenkins pipeline automatically:

1. Clone repo
2. Install dependencies
3. Run unit tests
4. Build Docker image
5. Push to Docker Hub
6. Deploy to Kubernetes

> Setup: Add your DockerHub credentials to Jenkins as `dockerhub-creds`.

---

### 4. Kubernetes Deployment

```bash
# Start Minikube
minikube start

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods
kubectl get services

# Get app URL
minikube service disaster-tweet-service --url
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📡 API Endpoints

| Method | Endpoint   | Description |
|--------|------------|-------------|
| GET    | `/`        | Browser UI  |
| GET    | `/health`  | Health check |
| POST   | `/predict` | Classify a tweet |

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tweet": "Earthquake hits the city, thousands displaced"}'
```

### Example Response

```json
{
  "tweet": "Earthquake hits the city, thousands displaced",
  "label": 1,
  "confidence": 0.9342,
  "result": "DISASTER"
}
```
