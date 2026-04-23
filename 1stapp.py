from dotenv import load_dotenv
import os
from openai import OpenAI
import numpy as np
import faiss 

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chat_history = []

# Step 1: Read + chunk files
documents = []
filenames = []

for filename in os.listdir("data"):
    with open(f"data/{filename}", "r") as file:
        content = file.read()

        sentences = content.split(".")
        chunk_size = 2

        for i in range(0, len(sentences), chunk_size):
            chunk = ".".join(sentences[i:i+chunk_size])

            if chunk.strip() != "":
                documents.append(chunk)
                filenames.append(filename)

# Step 2: Create embeddings for documents
doc_embeddings = []

for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    doc_embeddings.append(response.data[0].embedding)
    # Step 3: Convert to numpy array
embeddings_array = np.array(doc_embeddings).astype("float32")

# Step 4: Create FAISS index
index = faiss.IndexFlatL2(embeddings_array.shape[1])
index.add(embeddings_array)

while True:
    question = input("Ask something (type 'exit' to stop): ")

    if question.lower() == "exit":
        break

    context_question = question

# Take last 2 user questions for better context
    if len(chat_history) >= 4:
     context_question = (
        chat_history[-4]["content"] + " " +
        chat_history[-2]["content"] + " " +
        question
    )
    elif len(chat_history) >= 2:
     context_question = chat_history[-2]["content"] + " " + question

    question_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=context_question
  ).data[0].embedding

    question_vector = np.array([question_embedding]).astype("float32")
    distances, indices = index.search(question_vector, 5)

    selected_knowledge = ""

    for i, dist in zip(indices[0], distances[0]):
        if dist < 1.5:
            selected_knowledge += documents[i] + " "
    if selected_knowledge == "":
     selected_knowledge = documents[indices[0][0]]

    # ✅ Build messages FIRST
    messages = [
        {"role": "system", "content": "Answer only from the provided knowledge."}
    ]

    # Add past conversation
    for chat in chat_history:
        messages.append(chat)

    # Add current question
    messages.append({
        "role": "user",
        "content": f"Knowledge:\n{selected_knowledge}\n\nQuestion: {question}"
    })

    # ✅ NOW call API
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )

    answer = response.choices[0].message.content
    print(answer)

    # Save memory
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})