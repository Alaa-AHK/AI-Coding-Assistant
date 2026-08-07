import streamlit as st
import re
from src.cellula_core import CodingAssistant

st.set_page_config(
    page_title='Cellula AI Coding Assistant',
    page_icon='🤖',
    layout='wide'
)

# Custom styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stCodeBlock {
        background-color: #1E1E1E !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Settings")
    hf_token = st.text_input("Hugging Face API Token", type="password")
    model_name = st.text_input("Model Name", value="mistralai/Mistral-7B-Instruct-v0.3")
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        if 'assistant' in st.session_state:
            st.session_state.assistant.memory.clear()
        st.session_state.last_code = None
        st.session_state.needs_feedback = False
        st.session_state.last_query = ""
        st.rerun()
        
    st.divider()
    
    if 'assistant' in st.session_state:
        st.subheader("📚 Knowledge Base")
        count = st.session_state.assistant.vector_store.get_collection_count()
        st.metric("Documents", count)
        
        with st.expander("🧠 Memory Visualization"):
            for msg in st.session_state.assistant.memory.get_messages():
                st.text(f"{msg['role']}: {msg['content'][:50]}...")
                
    st.divider()
    st.subheader("📁 Upload File")
    uploaded_file = st.file_uploader("Upload .py or .txt", type=["py", "txt"])
    if uploaded_file is not None:
        file_content = uploaded_file.getvalue().decode("utf-8")
        st.session_state.uploaded_content = f"File {uploaded_file.name} content:\n```\n{file_content}\n```"
        st.success("File uploaded! Click the button below to add to chat.")
        if st.button("Add to Chat"):
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": st.session_state.uploaded_content})
            st.rerun()

st.title("🤖 Cellula AI Coding Assistant")
st.markdown("Your intelligent coding companion powered by Hugging Face and ChromaDB.")

if not hf_token:
    st.warning("Please enter your Hugging Face API token in the sidebar to continue.")
    st.stop()

if 'assistant' not in st.session_state or st.session_state.get('hf_token') != hf_token or st.session_state.get('model_name') != model_name:
    try:
        st.session_state.assistant = CodingAssistant(hf_token=hf_token, model_name=model_name)
        st.session_state.hf_token = hf_token
        st.session_state.model_name = model_name
    except Exception as e:
        st.error(f"Failed to initialize assistant: {str(e)}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_code" not in st.session_state:
    st.session_state.last_code = None
if "needs_feedback" not in st.session_state:
    st.session_state.needs_feedback = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Sync messages from session state to assistant memory if needed
if not st.session_state.assistant.memory.get_messages() and st.session_state.messages:
    for msg in st.session_state.messages:
        st.session_state.assistant.memory.add_message(msg['role'], msg['content'])

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Check if assistant message has code
        if message["role"] == "assistant":
            code_match = re.search(r'```(?:python)?\n(.*?)\n```', message["content"], re.DOTALL)
            if code_match:
                code = code_match.group(1)
                st.session_state.last_code = code
                if st.button("Run Code ▶", key=f"run_{i}"):
                    with st.spinner("Executing..."):
                        result = st.session_state.assistant.execute_code(code)
                        with st.expander("Execution Results", expanded=True):
                            if result['success']:
                                st.success("Success!")
                                if result['stdout']:
                                    st.code(result['stdout'])
                            else:
                                st.error("Error!")
                                if result['stderr']:
                                    st.code(result['stderr'])

if st.session_state.needs_feedback:
    st.info("I couldn't find relevant information to generate code. Please provide a solution so I can learn!")
    solution = st.text_area("Your solution:")
    if st.button("Submit Solution"):
        msg = st.session_state.assistant.provide_feedback(solution, st.session_state.last_query)
        st.session_state.messages.append({"role": "user", "content": f"Feedback provided: {solution}"})
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.session_state.needs_feedback = False
        st.rerun()
else:
    if query := st.chat_input("Ask a coding question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        st.session_state.last_query = query
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.assistant.process_query(query)
                st.markdown(result['response'])
                st.session_state.messages.append({"role": "assistant", "content": result['response']})
                
                if result['code']:
                    st.session_state.last_code = result['code']
                    if st.button("Run Code ▶", key=f"run_latest"):
                        with st.spinner("Executing..."):
                            exec_result = st.session_state.assistant.execute_code(result['code'])
                            with st.expander("Execution Results", expanded=True):
                                if exec_result['success']:
                                    st.success("Success!")
                                    if exec_result['stdout']:
                                        st.code(exec_result['stdout'])
                                else:
                                    st.error("Error!")
                                    if exec_result['stderr']:
                                        st.code(exec_result['stderr'])
                                        
                if result.get('needs_feedback'):
                    st.session_state.needs_feedback = True
                    st.rerun()
