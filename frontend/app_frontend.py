import streamlit as st
import requests
import json
import os

st.set_page_config(
    page_title="Assistente RAG Helpdesk",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "https://vinimoreira-rag-backend.hf.space/query" 

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso te ajudar com suas dúvidas de TI?"}]

with st.sidebar:
    st.title("🛠️ Configurações")
    top_k_slider = st.slider(
        "Documentos para busca (Top K):",
        min_value=1, max_value=10, value=5,
        help="Define quantos documentos relevantes serão usados para gerar a resposta."
    )
    st.markdown("---")
    if st.button("🗑️ Limpar Histórico da Conversa"):
        st.session_state.messages = [{"role": "assistant", "content": "Conversa reiniciada. Qual sua nova dúvida?"}]
        st.rerun()
    st.markdown("---")
    st.info("Projeto construído com FastAPI, LangChain e Streamlit.")

st.title("🤖 Assistente RAG para Helpdesk")
st.markdown("Este assistente responde perguntas com base em uma base de conhecimento de TI.")

def handle_query(user_prompt):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.messages.append({"role": "assistant", "content": ""})
    
    process_api_response(user_prompt)

def process_api_response(user_prompt):
    with st.spinner("Analisando dataset... 🧠"):
        try:
            payload = {"query": user_prompt, "top_k": top_k_slider}
            response = requests.post(API_URL, json=payload, timeout=120)
            response.raise_for_status() # Lança um erro para status code ruim

            data = response.json()
            answer = data.get("answer", "Não foi possível obter uma resposta.")
            sources = data.get("source_chunks", [])
            
            final_response_with_sources = answer
            if sources:
                unique_sources = sorted(list(set(os.path.basename(s['source']) for s in sources if s.get('source'))))
                if unique_sources:
                    final_response_with_sources += "\n\n---\n*Fontes consultadas:*\n"
                    final_response_with_sources += "\n".join(f"- `{s}`" for s in unique_sources)
            
            st.session_state.messages[-1] = {"role": "assistant", "content": final_response_with_sources}

        except requests.exceptions.RequestException as e:
            error_message = f"Erro de conexão com o back-end: {e}"
            st.session_state.messages[-1] = {"role": "assistant", "content": error_message}
        except Exception as e:
            error_message = f"Ocorreu um erro inesperado: {e}"
            st.session_state.messages[-1] = {"role": "assistant", "content": error_message}
    
    st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("---")
st.caption("Sem ideias? Tente um dos exemplos abaixo:")
col1, col2, col3 = st.columns(3)
example_questions = [
    "Como liberar espaço em disco no Windows?",
    "Minha impressora de rede não funciona, o que fazer?",
    "Como remover um virus do PC?",
]
if col1.button(example_questions[0], use_container_width=True):
    handle_query(example_questions[0])

if col2.button(example_questions[1], use_container_width=True):
    handle_query(example_questions[1])

if col3.button(example_questions[2], use_container_width=True):
    handle_query(example_questions[2])

if prompt := st.chat_input("Digite sua pergunta aqui..."):
    handle_query(prompt)