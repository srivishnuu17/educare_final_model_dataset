
# Ollama Integration – EduCareer AI

## Overview
EduCareer AI uses Ollama to run a locally hosted language model for generating conversational responses.

## Model Details
- Runtime: Ollama
- Model: llama3.2:1b
- Model size: Approximately 1.3 GB
- API endpoint: http://localhost:11434/api/chat

## Installation
Install Ollama from:
https://ollama.com/download

## Download the Model
```bash
ollama pull llama3.2:1b
```

## Run the Model
```bash
ollama run llama3.2:1b
```

## Start Ollama Server
```bash
ollama serve
```

## Integration
The EduCareer AI Streamlit application communicates with Ollama through its local REST API.

The application uses the Ollama chat endpoint to generate responses to general user queries.

## Purpose
Ollama provides the conversational response-generation component of the EduCareer AI system.
