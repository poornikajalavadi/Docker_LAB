# Lab 1: Dockerizing an MNIST Digit Classification Model

## Project Description
This lab demonstrates how to containerize a machine learning application using Docker. The project packages a pre-trained MNIST handwritten digit classifier into a Docker container, enabling reproducible and portable model inference across any environment.

## How It Works
1. `load_model.py` loads the pre-trained MNIST model (`mnist_model.pkl`) using joblib
2. If the model file is not found in the temp directory, it falls back to the local project directory
3. Once loaded, the script runs a sample prediction using the `sklearn.datasets` digits dataset
4. It prints the model type, model details, and compares the predicted vs actual label

## Prerequisites
- Docker installed on your machine
- Git (to clone the repository)

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/poornikajalavadi/Docker_LAB.git
cd Docker_LAB
```

### Build the Docker Image
```bash
docker build -t lab1:v1 .
```

### Run the Container
```bash
docker run lab1:v1
```

### Save the Docker Image
```bash
docker save lab1:v1 > my_image.tar
```

### Load a Saved Image
```bash
docker load < my_image.tar
```

## Docker Image
![Docker Images](images/docker_images.png)

## Expected Output
