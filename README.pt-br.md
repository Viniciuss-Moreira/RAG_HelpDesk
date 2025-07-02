[🇺🇸 View in English](./README.md)
---
# 🤖 Synapse Desk

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-F79533?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-20.10-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**Link para a Aplicação:** [Acesse o assistente de Helpdesk Aqui](https://vinimoreira-rag-frontend.hf.space)

---

![GIF da Aplicação](./img/demo.gif)

---

## 📖 Descrição do Projeto

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) de ponta a ponta, projetado para atuar como um assistente especialista para um helpdesk de TI. A aplicação é capaz de responder a perguntas em português com base em uma base de conhecimento customizada, utilizando técnicas avançadas para garantir a precisão e a relevância das respostas.

O sistema completo é implantado como uma arquitetura de microsserviços na nuvem, com um back-end de API robusto e um front-end interativo.

---

## 🛠️ Arquitetura do Sistema

O fluxo de dados e processamento segue a arquitetura RAG moderna:

1.  **Ingestão de Dados:** Documentos (`.txt`, `.pdf`, etc.) são carregados e processados em um pipeline offline. Eles são divididos em `chunks` e vetorizados usando o modelo de embedding `BAAI/bge-m3`.
2.  **Armazenamento Vetorial:** Os vetores e metadados são armazenados em um índice FAISS, otimizado para buscas de similaridade rápidas.
3.  **API (Back-end):** Uma API FastAPI, containerizada com Docker, recebe a pergunta do usuário.
4.  **Pipeline RAG de Inferência:**
    * **Busca Híbrida:** A pergunta é usada para realizar uma busca vetorial (densa) e uma busca por palavras-chave (esparsa, com TF-IDF).
    * **Re-ranking:** Os resultados das buscas são combinados e reordenados por um modelo Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) para selecionar os trechos de contexto mais relevantes.
    * **Geração:** O contexto refinado e a pergunta original são enviados para um LLM de ponta (`Mixtral-8x7B-Instruct`) através de um prompt otimizado para gerar a resposta final.
5.  **Interface (Front-end):** Uma aplicação Streamlit fornece uma interface de chat interativa, que consome a API de back-end e exibe as respostas e fontes em tempo real.

---

## ✨ Funcionalidades Principais

* **Busca Híbrida:** Combina o poder da busca semântica (vetores) com a precisão da busca por palavras-chave (TF-IDF).
* **Re-ranking de Contexto:** Aumenta drasticamente a relevância do contexto antes de ser enviado ao LLM, melhorando a qualidade das respostas.
* **Streaming de Respostas:** A interface de chat exibe as respostas palavra por palavra, como nos assistentes de IA modernos.
* **API Robusta:** Back-end construído com FastAPI, seguindo as melhores práticas.
* **Deploy em Nuvem:** Sistema completo implantado no Hugging Face Spaces, com o back-end utilizando hardware de GPU (via Docker) e o front-end em um ambiente CPU leve.

---

## 🚀 Stack de Tecnologias

* **Back-end:** Python, FastAPI, Uvicorn
* **Front-end:** Streamlit
* **IA & Machine Learning:** LangChain, PyTorch, Sentence Transformers, FAISS, Scikit-learn, Hugging Face (Hub, InferenceClient)
* **Deploy & Infraestrutura:** Docker, Hugging Face Spaces

---

## ⚙️ Como Executar Localmente

1.  **Pré-requisitos:**
    * Python 3.11+
    * Git

2.  **Clonar o Repositório:**
    ```bash
    git clone https://github.com/Viniciuss-Moreira/Synapse_Desk.git
    cd Synapse_Desk
    ```

3.  **Configurar o Ambiente Virtual:**
    ```bash
    python -m venv .venv
    # No Windows:
    .\.venv\Scripts\activate
    # No Mac/Linux:
    source .venv/bin/activate
    ```

4.  **Instalar Dependências:**
    * Para usar GPU, primeiro instale a versão correta do PyTorch.
    * Depois, instale o resto:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configurar Variáveis de Ambiente:**
    * Renomeie o arquivo `.env.example` para `.env`.
    * Adicione sua chave de API da Hugging Face: `HUGGINGFACE_API_TOKEN="hf_..."`

6.  **Executar a Ingestão de Dados:**
    * Adicione seus arquivos de conhecimento na pasta `data/raw`.
    * Rode o script de ingestão:
    ```bash
    python -m src.ingestion.embedder
    ```

7.  **Iniciar os Servidores (em dois terminais separados):**
    * **Terminal 1 (Back-end):**
      ```bash
      python -m uvicorn api.main:app
      ```
    * **Terminal 2 (Front-end):**
      ```bash
      streamlit run frontend/app_frontend.py
      ```

---

## 🔮 Possíveis Melhorias Futuras

* Testes com diferentes LLMs (GPT-4, Llama 3).
* Adição de um banco de dados para salvar o histórico de conversas e o feedback dos usuários.
