# KB Assistant Challenge

## Objective

Your task is to build a system that enables users to query contextual information from the script of the movie The Matrix. The system should retrieve relevant excerpts from the script and use them to help an AI agent generate accurate, context-aware responses.

You may use a Retrieval-Augmented Generation (RAG) approach, or any alternative design that effectively combines retrieval and generation to produce grounded answers. The focus is on building a solution that demonstrates strong retrieval capabilities and uses that context effectively in AI-driven responses.

## Challenge Goals

This challenge is divided into two parts:

-   Part 1 (Mandatory): Completing this part is required for your submission to be considered complete.
-   Part 2 (Optional but Recommended): This part is not required but will demonstrate deeper reasoning, richer retrieval, and more advanced capabilities.

### Part 1 - Core Functionality (Mandatory)

Your system must be able to answer basic factual queries based on the provided script of **The Matrix**.

Example queries:

-   Under what circumstances does Neo see a white rabbit?
-   How did Trinity and Neo first meet?
-   Why is there no sunlight in the future?
-   Who needs solar power to survive?
-   Why do the Agents want to capture Morpheus?
-   Describe the Nebuchadnezzar.
-   What is Nebuchadnezzar's crew made up of?

### Part 2 - Advanced Capabilities (Optional)

This part evaluates your system's ability to handle complex, composed, and reasoning-based queries.

Example queries:

-   How many times does Morpheus mention that Neo is the One?
-   Why are humans similar to a virus? And who says that?
-   Describe Cypher's personality.
-   What does Cypher offer to the Agents, and in exchange for what?
-   What is the purpose of the human fields, and who created them?

#### Recommendations (Optional)

To help you get started, here are some suggestions and prebuilt components available in this project. These are not required but may help you complete the challenge more efficiently and effectively:

-   **Script Loader**

    A custom loader is provided to parse and load The Matrix script. You can use this loader as-is or modify it as needed.

    See: [notebooks/01-loaders/01-matrix-script-loader.ipynb](notebooks/01-loaders/01-matrix-script-loader.ipynb)

-   **Retriever**

    We recommend using a **Qdrant-based** retriever.

    See: [notebooks/02-retriever/01-qdrant-retriever.ipynb](notebooks/02-retriever/01-qdrant-retriever.ipynb)

-   **LLM Agent**

    For implementing the AI agent, we recommend using **Pydantic-AI**.

    See: [notebooks/03-llm-agents/01-llm-agents.ipynb](notebooks/03-llm-agents/01-llm-agents.ipynb)

-   **Advanced Capability - MCP Server**

    To handle advanced reasoning and agent orchestration, especially for the requirements in **Part 2** of this challenge, we recommend using an **MCP server**. The environment already includes **Pydantic-AI**, which has built-in support for the MCP protocol.

    See: [https://ai.pydantic.dev/mcp/](https://ai.pydantic.dev/mcp/)

#### System Evaluation

As part of building a robust system, you should carefully consider:

-   **How will you evaluate the system?**

    What metrics or criteria will you use to assess the quality and accuracy of the responses?

-   **How will you ensure the agent does not hallucinate or rely on prior knowledge of the movie?**

    Your system should be designed to **only answer based on the retrieved context**, not the agent's pretrained knowledge.

## Environment Setup

_This setup is highly recommended but not obligatory. Work on the challenge using the environment of your preference._

### Prerequisites

-   Install Make:

    ```bash
    sudo apt install make
    ```

-   Install Docker following the official [Docker installation guide](https://docs.docker.com/engine/install/ubuntu/).

### Dev Container (Recommended)

To ensure a consistent development environment, this project uses a preconfigured Dev Container.

-   Open this repository in VS Code:
    ```bash
    code .
    ```
-   After installing the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, press F1 to open the Command Palette, type _Dev Containers_, and select: **Reopen in Container**

### Jupyter

Jupyter is preconfigured inside the Dev Container.
You can explore examples in the [notebooks/](notebooks/) directory.
When opening a notebook, select the appropriate kernel in the top-right corner: **Python Environments -> Python 3.12 (Global Env)**

### Custom Python Library

A local Python package named **kbac** (short for KB Assistant Challenge) is included in the environment. It contains utility functions to help you work with the project. You are encouraged to extend this library as needed. Example usage can be found in: [notebooks/01-loaders/01-matrix-script-loader.ipynb](notebooks/01-loaders/01-matrix-script-loader.ipynb). After you add to or modify this library, it is not necessary to rebuild the container. However, if you are using it in a Jupyter notebook, you should restart that notebook.

### Python Dependencies

You can install additional Python libraries by adding them to the **requirements.txt**. You should rebuild the container afterward (F1 + Rebuild Container).

### Environment Variables

You can define environment variables (such as `OPENAI_API_KEY`) in a `.env` file placed at the root of the project. These variables will be automatically loaded into the environment inside the Dev Container.

**Example `.env` file:**

```env
OPENAI_API_KEY=your-key-here
MY_CUSTOM_VAR=some-value
```
