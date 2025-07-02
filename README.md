# 🤖 Synapse Desk

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-F79533?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-20.10-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**Live Application Link:** [Access the Helpdesk assistent Here](https://vinimoreira-rag-frontend.hf.space)
![GIF of the Application](https://docs.streamlit.io/_images/st.chat_input_demo.gif)

---

## 📖 Project Description

This project implements a complete, end-to-end Retrieval-Augmented Generation (RAG) system designed to act as an expert IT helpdesk assistant. The application is capable of answering questions in Portuguese based on a custom knowledge base, utilizing advanced techniques to ensure the accuracy and relevance of its responses.

The entire system is deployed as a microservices architecture in the cloud, featuring a robust API backend and an interactive user interface.

---

## 🛠️ System Architecture

The data and processing flow follows a modern RAG architecture:

1.  **Data Ingestion:** Source documents (`.txt`) are loaded and processed in an offline pipeline. They are split into chunks and vectorized using the `BAAI/bge-m3` embedding model.
2.  **Vector Storage:** The resulting vectors and metadata are stored in a FAISS index, optimized for high-speed similarity searches.
3.  **API Backend:** A FastAPI API, containerized with Docker, receives user queries.
4.  **RAG Inference Pipeline:**
    * **Hybrid Search:** The query is used to perform both a dense vector search and a sparse keyword search (TF-IDF).
    * **Re-ranking:** The results from the search stage are combined and re-ordered by a Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to select the most relevant context snippets.
    * **Generation:** The refined context and the original query are fed into a state-of-the-art LLM (`Mixtral-8x7B-Instruct`) through an optimized prompt to generate the final answer.
5.  **UI Frontend:** A Streamlit application provides an interactive chat interface, which consumes the backend API and displays the responses and their sources.

---

## ✨ Key Features

* **Advanced RAG Pipeline:** Goes beyond basic RAG with a more sophisticated retrieval process.
* **Hybrid Search:** Combines the semantic power of vector search with the precision of keyword-based search.
* **Context Re-ranking:** Dramatically increases the relevance of the context provided to the LLM, improving response quality.
* **Streaming Responses:** The chat interface displays answers token-by-token for a modern user experience.
* **Microservices Architecture:** Decoupled front-end and back-end, deployed independently, which is a professional standard.
* **Cloud Deployment:** Fully deployed on Hugging Face Spaces, leveraging CPU for the UI and a dedicated environment for the API.

---

## 🚀 Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** Streamlit
* **AI & Machine Learning:** LangChain, PyTorch, Sentence Transformers, FAISS, Hugging Face (Hub, InferenceClient)
* **Deployment & Infrastructure:** Docker, Hugging Face Spaces, Git

---

## ⚙️ How to Run Locally

1.  **Prerequisites:**
    * Python 3.11+
    * Git

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Viniciuss-Moreira/Synapse_Desk.git
    cd Synapse_Desk
    ```

3.  **Set Up the Virtual Environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On Mac/Linux:
    source .venv/bin/activate
    ```

4.  **Install Dependencies:**
    * To use a local GPU, first install the correct PyTorch version from their official website.
    * Then, install the remaining requirements:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    * Rename the `.env.example` file to `.env`.
    * Add your Hugging Face API key: `HUGGINGFACE_API_TOKEN="hf_..."`

6.  **Run the Data Ingestion:**
    * Add your knowledge base files to the `data/raw` folder.
    * Run the ingestion script:
    ```bash
    python -m src.ingestion.embedder
    ```

7.  **Start the Servers (in two separate terminals):**
    * **Terminal 1 (Backend API):**
      ```bash
      python -m uvicorn api.main:app
      ```
    * **Terminal 2 (Frontend UI):**
      ```bash
      streamlit run frontend/app_frontend.py
      ```

---

## 🔮 Future Improvements

* Experimentation with other LLMs (like GPT-4o or Llama 3).
* Addition of a database to store chat history and user feedback.
