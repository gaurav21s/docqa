# Documentation QA System

This project implements a Question Answering system for NVIDIA CUDA documentation. It includes web crawling, data preprocessing, retrieval, re-ranking, and question answering components, all integrated into a Streamlit-based user interface.

## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Troubleshooting](#troubleshooting)
7. [Contributing](#contributing)

## Features

- Web crawling of NVIDIA CUDA documentation
- Advanced data preprocessing and chunking
- Hybrid retrieval system combining BM25 and DPR
- Re-ranking of retrieved documents
- Question answering using the LLM
- User-friendly Streamlit interface

## Prerequisites

- Python 3.8 or higher
- Docker (for running Milvus)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/docqa.git
   cd docqa
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up Milvus:
   - Follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md) to set up Milvus on your system.
   - Ensure Milvus is running before using the system.
   - Milvus provides an installation script to install it as a docker container. The script is available in the Milvus repository. To install Milvus in Docker, just run

   ```
   # Download the installation script
   $ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

   # Start the Docker container
   $ bash standalone_embed.sh start
   ```

5. Set up the Groq API key:
   - Sign up for a Groq account and obtain your API key.
   - Set the API key as an environment variable:
     ```
     export GROQ_API_KEY=your_api_key_here
     ```
   - For Windows, use:
     ```
     setx GROQ_API_KEY your_api_key_here
     ```
   - Alternatively, you can add this line to your `.bashrc` or `.zshrc` file (for macOS/Linux) to make it permanent.
   

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Using the interface:
   - Click "Crawl and Process Data" to fetch and process the latest NVIDIA CUDA documentation (this may take some time).
   - Once processing is complete, enter your question about CUDA in the text input field.
   - Click "Get Answer" to retrieve relevant information and generate an answer.
   - View the answer and the top relevant documents, including their titles and URLs.

4. To clear the database and start fresh:
   - Click the "Clear Database" button in the Streamlit interface.

## Project Structure

- `app.py`: Main Streamlit application file
- `utils/web_crawl.py`: Web crawler for NVIDIA CUDA documentation using Scrapy
- `utils/data.py`: Data preprocessing, chunking, and Milvus database management
- `utils/retrieve.py`: Hybrid document retrieval (BM25 + DPR) and re-ranking
- `utils/qna.py`: Question answering system using Groq LLM
- `utils/requirements.txt`: List of Python dependencies
- `utils/logger.py`: Logging utility for the application

## Troubleshooting

- If you encounter issues with Milvus, ensure that Docker is running and that the Milvus container is active.
- For "module not found" errors, make sure you've activated the virtual environment and installed all dependencies.
- If you experience out-of-memory errors, try reducing the `chunk_size` in `data.py` or use a machine with more RAM.
- For any Groq API-related issues, verify that your API key is correctly set and that you have an active subscription.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

