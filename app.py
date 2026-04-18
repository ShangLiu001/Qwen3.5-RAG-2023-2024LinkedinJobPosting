import streamlit as st
import ollama

with st.sidebar:
    your_name = st.text_input("What's your name?")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Qwen_logo_2024.svg/240px-Qwen_logo_2024.svg.png", width=120)
    st.caption("🖥️ Running Qwen3.5 on NVIDIA L4 via Ollama")
    "[View the source code](https://github.com/kni-neu/serving-llama)"

if your_name:
    st.title("Hi there, " + your_name + "! 👋")
else:
    st.title("My Very Own Chatbot 💬")

st.caption("🚀 A Streamlit chatbot powered by Qwen3.5")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = ollama.chat(model='qwen3.5', messages=[{'role': 'user', 'content': prompt}])['message']['content']
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
