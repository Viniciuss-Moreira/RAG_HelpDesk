import streamlit as st
import requests
import os

st.set_page_config(page_title="Assistente RAG", page_icon="🤖", layout="centered")
st.title("🤖 Assistente de Helpdesk (RAG)")

API_URL = "http://127.0.0.1:8000/query"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso te ajudar com suas dúvidas de TI?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Analisando a base de conhecimento..."):
            try:
                payload = {"query": prompt, "top_k": 5}
                response = requests.post(API_URL, json=payload, timeout=120)

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Não foi possível obter uma resposta.")
                    sources = data.get("source_chunks", [])
                    
                    full_response = answer
                    if sources:
                        unique_sources = set(chunk['source'] for chunk in sources)
                        full_response += "\n\n---\n*Fontes consultadas:*"
                        for source_file in unique_sources:
                            full_response += f"\n- `{os.path.basename(source_file)}`"
                    
                    message_placeholder.markdown(full_response)
                else:
                    full_response = f"Erro da API: {response.status_code} - {response.text}"
                    message_placeholder.error(full_response)

            except requests.exceptions.RequestException as e:
                full_response = f"Erro de conexão com o back-end: {e}"
                message_placeholder.error(full_response)
                
    st.session_state.messages.append({"role": "assistant", "content": full_response})