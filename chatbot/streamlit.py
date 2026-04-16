import streamlit as st
import requests

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Swift Ship Chatbot", layout="wide")

st.title("🚚 Swift Ship Chatbot")
st.markdown("Ask questions about shipments, tracking, and support.")

# -----------------------------
# SIDEBAR (SYSTEM INFO)
# -----------------------------
st.sidebar.header("System Info")

if st.sidebar.button("Check API Health"):
    try:
        res = requests.get(f"{API_URL}/health")
        if res.status_code == 200:
            data = res.json()
            st.sidebar.success("API is healthy ✅")
            st.sidebar.write(data)
        else:
            st.sidebar.error("API error")
    except:
        st.sidebar.error("Cannot connect to API")

# -----------------------------
# CHAT INPUT
# -----------------------------
question = st.text_input("Enter your question:")

top_k = st.slider("Top-K Sources", 1, 10, 5)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                        "top_k": top_k
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    # -----------------------------
                    # ANSWER
                    # -----------------------------
                    st.subheader("💬 Answer")
                    st.write(data["answer"])

                    # -----------------------------
                    # SOURCE INFO
                    # -----------------------------
                    st.subheader("📚 Sources")
                    if data["sources"]:
                        for i, src in enumerate(data["sources"]):
                            with st.expander(f"Source {i+1} (score: {src['score']})"):
                                st.write(src["text"])
                    else:
                        st.info("No sources found.")

                else:
                    st.error(f"Error: {response.text}")

            except Exception as e:
                st.error(f"Failed to connect: {e}")