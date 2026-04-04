pipeline {
    agent any

    environment {
        // Replace with your Docker Hub username
        DOCKER_IMAGE = "rohanxmudrale28/disaster-tweet-classifier"
        DOCKER_TAG   = "latest"
    }

    stages {

        stage('1 - Clone Repository') {
            steps {
                echo '📥 Cloning repository from GitHub...'
                git branch: 'main',
                    url: 'https://github.com/rohanxmudrale28/Disaster-Tweet-Classifier'
            }
        }

        stage('2 - Install Dependencies') {
            steps {
                echo '📦 Installing Python dependencies...'
                sh 'pip install -r requirements.txt'
            }
        }

        stage('3 - Run Tests') {
            steps {
                echo '🧪 Running unit tests...'
                sh 'python -m pytest tests/ -v --tb=short'
            }
        }

        stage('4 - Build Docker Image') {
            steps {
                echo '🐳 Building Docker image...'
                sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
            }
        }

        stage('5 - Push to Docker Hub') {
            steps {
                echo '🚀 Pushing image to Docker Hub...'
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh """
                        docker login -u \$DOCKER_USER -p \$DOCKER_PASS
                        docker push ${DOCKER_IMAGE}:${DOCKER_TAG}
                        docker logout
                    """
                }
            }
        }

        stage('6 - Deploy to Kubernetes') {
            steps {
                echo '☸️  Deploying to Kubernetes...'
                sh 'kubectl apply -f k8s/deployment.yaml'
                sh 'kubectl apply -f k8s/service.yaml'
                sh 'kubectl rollout status deployment/disaster-tweet-classifier'
            }
        }

    }

    post {
        success {
            echo '✅ Pipeline completed successfully! App is live.'
        }
        failure {
            echo '❌ Pipeline failed. Check the logs above.'
        }
        always {
            echo '🧹 Cleaning up local Docker images...'
            sh "docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || true"
        }
    }
}
