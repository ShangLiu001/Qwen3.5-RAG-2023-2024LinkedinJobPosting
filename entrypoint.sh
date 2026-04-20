#!/bin/bash
echo "Starting Ollama server..."
ollama serve &
sleep 5
echo "Starting Streamlit RAG app..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
