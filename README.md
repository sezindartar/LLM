# 🤖 LLM Portfolio - AI Projects

A comprehensive portfolio showcasing my work in Large Language Models (LLM) and Natural Language Processing. This repository contains examples ranging from fundamental ML concepts to advanced NLP applications.

## 🎯 Project Overview

This portfolio demonstrates practical applications of modern NLP techniques and LLM technologies through hands-on projects. Each component is designed to showcase different aspects of AI development, from basic concepts to production-ready systems.

## 📚 Core Components

### 🧠 Foundation LLM Concepts
- **Transformer Architecture**: Implementation and explanation of core transformer components
- **Tokenization**: Text preprocessing and tokenization techniques
- **Sentiment Analysis**: Basic emotion detection using pre-trained models
- **LLM Terminology**: Comprehensive guide to essential concepts

### 🔗 LangChain & Chain-of-Thought
- **Framework Integration**: Advanced problem-solving using LangChain
- **OpenAI API**: Seamless integration with GPT models
- **Reasoning Chains**: Implementation of complex multi-step reasoning

### 🎨 Multilingual Story Generation
- **Language Support**: Story generation in Turkish, English, German, French, and Spanish
- **Quality Assessment**: Automated story quality evaluation
- **Sentiment Analysis**: Emotion detection in generated content
- **Interactive Interface**: User-friendly story creation system

### 📊 Model Deployment with Gradio
- **BERT Integration**: Custom BERT-based chatbot for Turkish customer service
- **Web Interface**: Professional Gradio-powered web application
- **Real-time Processing**: Instant response generation

### 🔍 RAG (Retrieval Augmented Generation) System
- **FAISS Vector Store**: High-performance document indexing
- **PDF Processing**: Automatic document ingestion and processing
- **Natural Language Queries**: Conversational question-answering interface
- **Real-time Retrieval**: Instant information extraction from documents

### 🌐 AI Code Assistant (Dockerized)
- **FastAPI Backend**: High-performance API server
- **Streamlit Frontend**: Interactive web interface
- **Code Analysis**: Automated code review and suggestions
- **Security Scanning**: Built-in security vulnerability detection
- **Docker Ready**: One-command deployment

## 🛠️ Technology Stack

### Core Libraries
```
transformers, torch, langchain, openai, gradio, streamlit, fastapi
```

### ML/AI Components
- **Models**: GPT-4, BERT, RoBERTa
- **Vector Store**: FAISS
- **Embeddings**: OpenAI Embeddings
- **Languages**: Python, JavaScript

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key_here
```

### Docker Deployment (Recommended)
Run all applications with a single command:
```bash
docker-compose up -d
```

### Individual Module Usage
Navigate to specific modules and explore:
```bash
cd 01-LLM  # Start with fundamentals
python examples/tokenization_demo.py

cd ../04-LLM  # Try vector search
python semantic_search_example.py

cd ../07-LLM  # Deploy applications
docker-compose up -d
```

## ✨ Key Features

- **🌍 Multilingual Support**: Generate and analyze content in 5+ languages
- **🤖 Intelligent Chatbot**: Turkish-optimized customer service bot
- **📖 Document Q&A**: Query PDF documents using natural language
- **🔒 Code Security**: Automated security analysis for code projects
- **🚀 Easy Deployment**: Docker-containerized for seamless deployment

## 📁 Project Structure

```
├── 01_llm_fundamentals/     # Basic LLM concepts and implementations
├── 02_langchain_projects/   # LangChain framework applications
├── 03_story_generation/     # Multilingual story creation system
├── 04_gradio_deployment/    # Web-based chatbot deployment
├── 05_rag_system/          # Retrieval Augmented Generation
├── 06_ai_code_assistant/   # Dockerized code analysis tool
├── docker-compose.yml      # Multi-container deployment
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 📈 Learning Objectives

This portfolio is perfect for those looking to gain hands-on experience in:

- **Transformer Architecture**: Deep understanding of attention mechanisms
- **LangChain Framework**: Building complex AI workflows
- **RAG Systems**: Implementing retrieval-augmented generation
- **Multilingual NLP**: Working with multiple languages
- **Production Deployment**: Docker containerization and web deployment
- **API Integration**: OpenAI and other ML service integrations


