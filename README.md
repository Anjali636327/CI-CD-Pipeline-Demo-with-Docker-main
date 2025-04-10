# CI/CD Pipeline Demo with Docker

This repository demonstrates a simple CI/CD pipeline using **GitHub Actions** and Docker to automate testing and deployment for a Python project.

## Features
- Automates testing of Python code using **pytest**.
- Runs the pipeline on every push or pull request to the `main` branch.
- Builds and pushes a Docker image to Docker Hub as part of the CI/CD pipeline.

## Prerequisites
1. **Python**: Ensure Python 3.9 or later is installed.
2. **Git**: Ensure Git is installed and configured.
3. **Docker**: Ensure Docker is installed and running.
4. **GitHub Secrets**: Add the following secrets to your GitHub repository:
   - `DOCKER_USERNAME`: Your Docker Hub username.
   - `DOCKER_PASSWORD`: Your Docker Hub password.

Alternatively, you can use a `.env` file to store your Docker credentials locally. The `.env` file should look like this:
```plaintext
DOCKER_USERNAME=your-dockerhub-username
DOCKER_PASSWORD=your-dockerhub-password
```
The workflow reads these environment variables using the `source .env` command.

## Project Structure
```
ci-cd-demo/
├── app.py             # Python script with basic functionality
├── test_app.py        # Test script for app.py
├── requirements.txt   # List of project dependencies
├── Dockerfile         # Docker configuration file
└── .github/
    └── workflows/
        └── ci.yml    # GitHub Actions workflow configuration
```

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ci-cd-demo.git
   cd ci-cd-demo
   ```

2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run the application locally:
   ```bash
   python app.py
   ```

   Output:
   ```
   5 + 3 = 8
   ```

4. Run tests locally:
   ```bash
   pytest
   ```

   Example output:
   ```
   ===================== test session starts =====================
   collected 1 item

   test_app.py .                                             [100%]

   ===================== 1 passed in 0.02s =====================
   ```

## Docker Setup

### Build Docker Image
To containerize the application, build the Docker image using the provided `Dockerfile`:
```bash
docker build -t ci-cd-demo .
```

### Run Docker Container
Run the container using the built image:
```bash
docker run --name ci-cd-container ci-cd-demo
```

As the application contains a simple summation functionality, the container will exit after printing the output (e.g., `5 + 3 = 8`).

### Verify Running Containers
If you want to check running containers:
```bash
docker ps
```
If the container has exited, you can list all containers (including stopped ones):
```bash
docker ps -a
```

### Remove Container
After testing, you can clean up by removing the container:
```bash
docker rm ci-cd-container
```

### Push Image to Docker Hub
To push the image to Docker Hub, ensure you are logged in and run:
```bash
docker tag ci-cd-demo your-dockerhub-username/ci-cd-demo

docker push your-dockerhub-username/ci-cd-demo
```

## CI/CD Pipeline

### Workflow Configuration
The pipeline is defined in `.github/workflows/ci.yml`.

- **Trigger**: The pipeline runs on:
  - Push events to the `main` branch.
  - Pull request events targeting the `main` branch.

### Continuous Integration (CI)
- **Objective**: Automate the testing of code changes to ensure they do not break the application.
- **Steps in the Workflow**:
  1. **Checkout Code**: Retrieves the repository's code.
  2. **Set Up Python**: Configures the environment with the required Python version.
  3. **Install Dependencies**: Installs the required libraries listed in `requirements.txt`.
  4. **Run Tests**: Executes `pytest` to verify the functionality of the code.

### Continuous Deployment (CD)
- **Objective**: Automate the deployment of tested and approved code.
- **Steps in the Workflow**:
  1. **Log in to Docker Hub**: Uses secrets or `.env` file to authenticate.
  2. **Build Docker Image**: Creates a Docker image for the application.
  3. **Push Docker Image**: Pushes the Docker image to Docker Hub.

### Workflow File (`ci.yml`)
```yaml
name: CI/CD Pipeline with Docker

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      # Checkout the code
      - name: Checkout code
        uses: actions/checkout@v3

      # Set up Python
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      # Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # Run tests
      - name: Run tests
        run: pytest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      # Checkout the code
      - name: Checkout code
        uses: actions/checkout@v3

      # Load environment variables from .env
      - name: Load .env variables
        run: |
          set -a
          [ -f .env ] && . .env
          set +a

      # Log in to DockerHub
      - name: Log in to DockerHub
        run: |
          echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

      # Build and push the Docker image
      - name: Build and push Docker image
        run: |
          docker build -t ${{ secrets.DOCKER_USERNAME }}/ci-cd-demo:${{ github.sha }} .
          docker push ${{ secrets.DOCKER_USERNAME }}/ci-cd-demo:${{ github.sha }}
```

### Steps to Observe the Pipeline
1. Push changes to the `main` branch:
   ```bash
   git add .
   git commit -m "Updated project files"
   git push origin main
   ```

2. Go to the "Actions" tab in the GitHub repository.
3. Monitor the status of the workflow runs.

## Additional Notes
As the application in this repository contains a simple summation functionality, the Docker container will not run for long periods. This is suitable for demo purposes but can be extended for more complex applications.

## Why Use Both Local Testing and GitHub Actions?

In software development, ensuring code quality is critical. While **pytest** allows developers to test their code locally, automating tests with **GitHub Actions** adds an extra layer of reliability. This document explains why both approaches are essential and how they complement each other.

---

### Why Automate Testing if I Already Ran Pytest Locally?

#### 1. **Eliminates Human Error**
   - Developers might forget to run tests locally, especially in fast-paced environments or under time pressure.
   - GitHub Actions ensures that tests are **always run**, regardless of whether someone remembers to do it manually.

#### 2. **Consistent Testing Environment**
   - Your local setup (e.g., Python version, dependencies, OS) might differ from other developers' setups or the production environment.
   - GitHub Actions runs tests in a **clean, isolated environment** (e.g., Ubuntu with a specific Python version). This consistency ensures your code works beyond your machine.

   **Example:**
   - Locally, you use Python 3.9 and dependency version `1.2.3`.
   - Another team member or the production server uses Python 3.8 or dependency `1.3.0`.
   - Tests might pass locally but fail in those environments. GitHub Actions catches these discrepancies.

#### 3. **Team Collaboration**
   - In collaborative projects, multiple developers contribute code. You don’t know if your changes will break something when combined with others' work.
   - GitHub Actions runs tests **on the combined codebase**, ensuring everything works together.

   **Example:**
   - Developer A and Developer B both make changes. Each runs pytest locally and sees no issues.
   - But when their changes merge, they conflict, breaking the code. GitHub Actions catches this problem.

#### 4. **Integration Testing**
   - Local pytest typically focuses on testing specific changes.
   - Automated workflows might run **integration tests** or tests involving multiple components, which may not be run locally due to time constraints.

#### 5. **Accountability and Visibility**
   - Test results in GitHub Actions are visible to the entire team.
   - A failed test on a pull request alerts everyone, making the issue easier to spot and fix before merging.

#### 6. **Catch Edge Cases**
   - GitHub Actions can run tests in multiple configurations (e.g., Python 3.8, 3.9, 3.10).
   - Even if your local tests pass in Python 3.9, automation might reveal failures in Python 3.8 or another environment.

---

### What’s the Point of Running Pytest Locally, Then?

Running pytest locally serves as a **first line of defense**:
- **Fast Feedback**: Fix errors immediately while you’re coding.
- **Prevents Broken Code from Being Pushed**: Saves time by catching obvious issues before committing.

**GitHub Actions** is the **second line of defense**:
- Ensures your changes work in a clean, consistent, and collaborative environment.
- Verifies compatibility and integration with other developers’ work.
- Provides a record of test results tied to commits and pull requests.

---

### Analogy
- **Locally running pytest** = **You testing your homework** before turning it in.
- **GitHub Actions** = **The teacher grading your homework** to ensure it's correct and consistent for everyone.

---

## Conclusion
Using both **pytest locally** and **GitHub Actions** creates a reliable workflow:
- Locally, you catch and fix issues early, ensuring code quality before sharing it.
- GitHub Actions validates the code in a shared, consistent environment, safeguarding the integrity of the entire project.

By combining these practices, you minimize risks, ensure high-quality contributions, and streamline collaboration in software development.
