import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
import numpy as np
import faiss
from pypdf import PdfReader
from io import BytesIO
import pandas as pd

# -------------------- SESSION --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- API --------------------
load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------- UI --------------------
st.title("Smart Knowledge Assistant 🤖")
st.divider()

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

uploaded_files = st.file_uploader(
    "Upload files (TXT, PDF, Excel)",
    type=["txt", "pdf", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 0:
    st.success(f"{len(uploaded_files)} files uploaded")
    for f in uploaded_files:
        st.write("📄", f.name)
    st.info("Using uploaded files for answers")
else:
    st.info("Using default knowledge base")

# -------------------- CHAT HISTORY --------------------
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# -------------------- HELPER: BUILD INDEX --------------------
@st.cache_resource(show_spinner=False)
def build_index(documents):

    if not documents:
        return [], None

    # ✅ LIMIT documents
    MAX_DOCS = 1000
    documents = documents[:MAX_DOCS]

    doc_embeddings = []
    BATCH_SIZE = 100

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        doc_embeddings.extend([item.embedding for item in response.data])

    embeddings_array = np.array(doc_embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)

    return documents, index

# -------------------- DEFAULT DATA --------------------
@st.cache_resource
def load_data():
    documents = []

    for filename in os.listdir("Data"):
        with open(f"Data/{filename}", "r") as file:
            content = file.read()
            sentences = content.split(".")

            for i in range(0, len(sentences), 2):
                chunk = ".".join(sentences[i:i+2])
                if chunk.strip():
                    documents.append(f"[{filename}] {chunk}")

    return build_index(documents)

# -------------------- FILE PROCESSING --------------------
def process_uploaded_file(file_bytes, file_name):

    documents = []

    # -------- TXT --------
    if file_name.endswith(".txt"):
        content = file_bytes.decode("utf-8")

        sentences = content.split(".")
        for i in range(0, len(sentences), 2):
            chunk = ".".join(sentences[i:i+2])
            if chunk.strip():
                documents.append(f"[{file_name}] {chunk}")

    # -------- PDF --------
    elif file_name.endswith(".pdf"):
        file = BytesIO(file_bytes)
        reader = PdfReader(file)

        content = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text + " "

        sentences = content.split(".")
        for i in range(0, len(sentences), 2):
            chunk = ".".join(sentences[i:i+2])
            if chunk.strip():
                documents.append(f"[{file_name}] {chunk}")

    # -------- EXCEL --------
    elif file_name.endswith(".xlsx"):
        file = BytesIO(file_bytes)
        dfs = pd.read_excel(file, sheet_name=None)

        sheet_names = list(dfs.keys())

        # ✅ Persist selection
        key = f"sheets_{file_name}"

        if key not in st.session_state:
            st.session_state[key] = sheet_names

        selected_sheets = st.multiselect(
            f"Select sheets for {file_name}",
            sheet_names,
            default=st.session_state[key],
            key=key
        )

        for sheet in selected_sheets:
            df = dfs[sheet].fillna("")

            st.subheader(f"📊 Preview: {sheet}")
            st.dataframe(df.head())

            for _, row in df.iterrows():
                row_text = " | ".join([str(x) for x in row.values])
                if row_text.strip():
                    documents.append(f"[{file_name} | {sheet}] {row_text}")

            numeric_cols = df.select_dtypes(include=["number"])
            if not numeric_cols.empty:
                summary = numeric_cols.describe().to_string()
                documents.append(f"[{file_name} | {sheet}] NUMERIC SUMMARY: {summary}")

    # -------- UNKNOWN --------
    else:
        return [], None

    return documents, None

# -------------------- LOAD DATA --------------------
if uploaded_files and len(uploaded_files) > 0:
    all_documents = []

    for file in uploaded_files:
        docs, _ = process_uploaded_file(
            file.getvalue(),
            file.name
        )
        if docs:
            all_documents.extend(docs)

    # ✅ Empty upload fix
    if uploaded_files and not all_documents:
        st.error("Uploaded files contain no readable content.")
        st.stop()

    documents, index = build_index(all_documents)

else:
    documents, index = load_data()

# -------------------- DEBUG --------------------
if uploaded_files:
    st.success(f"📄 Using {len(uploaded_files)} uploaded files")
    st.write(f"Total chunks created: {len(documents)}")
else:
    st.write(f"📚 Using default data | Total chunks: {len(documents)}")

if not documents or index is None:
    st.warning("No documents available.")
    st.stop()

# -------------------- CHAT INPUT --------------------
question = st.chat_input("Ask something...")

if question:

    with st.chat_message("user"):
        st.write(question)

    history = st.session_state.chat_history

    # Context improvement
    context_question = question
    if len(history) >= 4:
        context_question = (
            history[-4]["content"] + " " +
            history[-2]["content"] + " " +
            question
        )
    elif len(history) >= 2:
        context_question = history[-2]["content"] + " " + question

    # Embedding
    question_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=context_question
    ).data[0].embedding

    question_vector = np.array([question_embedding]).astype("float32")

    # 🔍 RETRIEVAL
    if any(word in question.lower() for word in ["all", "list", "everything", "show all"]):
        selected_knowledge = " ".join(documents)
    else:
        k = min(10, len(documents))
        distances, indices = index.search(question_vector, k)
        selected_knowledge = " ".join([documents[i] for i in indices[0]])

    if selected_knowledge.strip() == "":
        selected_knowledge = documents[0]

    # -------------------- LLM --------------------
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant.

- Answer ONLY from the provided knowledge
- Use bullet points when useful
- Be clear and structured
- If not found, say: Not found in document
"""
        }
    ]

    for chat in history[-6:]:
        messages.append(chat)

    messages.append({
        "role": "user",
        "content": f"Knowledge:\n{selected_knowledge}\n\nQuestion: {question}"
    })

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )

    answer = response.choices[0].message.content

    # -------------------- OUTPUT --------------------
    with st.chat_message("assistant"):
        st.write(answer)

        with st.expander("📄 Sources used"):
            sources = set()
            for doc in selected_knowledge.split("["):
                if "]" in doc:
                    sources.add("[" + doc.split("]")[0] + "]")

            st.write("\n".join(sources))

    # Save history
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# -------------------- DOWNLOAD CHAT --------------------
chat_text = ""
for chat in st.session_state.chat_history:
    chat_text += f"{chat['role'].upper()}: {chat['content']}\n\n"

st.download_button(
    "⬇️ Download Chat",
    chat_text,
    file_name="chat.txt"
)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Controls")
    st.write(f"Documents loaded: {len(documents)}")