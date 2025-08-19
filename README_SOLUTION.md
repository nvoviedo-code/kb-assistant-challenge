# Matrix Knowledge Base Assistant

A powerful Retrieval-Augmented Generation (RAG) API that can answer questions about "The Matrix" movie script, handling both simple and complex queries with advanced reasoning capabilities.

## Features

- Document loading from PDF movie scripts
- Vector embedding and retrieval using Qdrant
- LLM-powered response generation using OpenAI
- Advanced reasoning for complex multi-step queries
- FastAPI-based REST endpoints

## Requirements

- Python 3.12+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key

## Installation

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nvoviedo-code/kb-assistant-challenge.git
   cd kb-assistant-challenge
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Configure your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```

5. Run the application:
   ```bash
   uvicorn src.app:app --reload
   ```

### Docker Setup

1. Make sure Docker and Docker Compose are installed on your system.

2. Configure your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```

3. Run the application using Make:
   ```bash
   make run-server
   ```

   This command will start both the API service and a Qdrant vector database in containers.

## Usage

Once the server is running, the API will be available at http://localhost:8000

> **Note:**  
> - To check the Qdrant dashboard, visit `http://localhost:6333/dashboard`.
> - To explore and test the API endpoints interactively, open the Swagger UI at `http://localhost:8000/docs` after starting the server.


## Running Tests

```bash
# Run all tests
pytest

# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/
```