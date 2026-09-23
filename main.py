from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import streamlit as st
import hashlib
import re

st.title("Chatbot PDF IA")

with st.sidebar:
    st.header("⚙️ Paramètres RAG")
    chunk_size = st.slider("Taille des chunks", 200, 1000, 500)
    chunk_overlap = st.slider("Chevauchement", 0, 200, 100)
    n_results = st.slider("Passages récupérés", 1, 10, 5)
    st.markdown("---")
    st.info("💡 **Chunks petits** = réponses précises\n\n💡 **Chunks grands** = plus de contexte")
    st.markdown("---")
    uploaded_files = st.file_uploader(      # 👈 déplacé dans la sidebar
        "📄 Choisis un ou plusieurs PDFs",
        type="pdf",
        accept_multiple_files=True
    )
    # Afficher les fichiers chargés
    if uploaded_files:
        st.success(f"📂 {len(uploaded_files)} fichier(s) chargé(s) :")
        for f in uploaded_files:
            st.caption(f"• {f.name}")
    st.markdown("---")
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.session_state.pop("collection", None)
        st.session_state.pop("current_hash", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Pose une question")

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

model = load_model()

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

chroma_client = get_chroma_client()

def get_files_hash(files):
    combined = b""
    for f in files:
        combined += f.read()
        f.seek(0)
    return hashlib.md5(combined).hexdigest()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\w\s.,;:!?\'\"()\-]', '', text)
    return text.strip()

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return clean_text(text)

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def embed_texts(texts):
    return model.encode(texts)

def create_chromadb_collection(embeddings, texts, collection_name):
    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing:
        return chroma_client.get_collection(name=collection_name)

    collection = chroma_client.create_collection(name=collection_name)
    for i, embedding in enumerate(embeddings):
        collection.add(
            ids=[str(i)],
            embeddings=[embedding.tolist()],
            documents=[texts[i]]
        )
    return collection

def query_chromadb_collection(collection, query, n_results=5):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results

def query_ollama(prompt):
    history = []
    for message in st.session_state.messages[:-1]:
        history.append({
            "role": message["role"],
            "content": message["content"]
        })
    history.append({"role": "user", "content": prompt})

    with st.spinner("🤔 Mistral réfléchit..."):
        return ollama.chat(
            model="mistral",
            messages=history
        )

def build_prompt(results, query):
    documents = results["documents"][0]
    context = "\n\n".join(documents)
    return f"""
Tu es un assistant IA spécialisé dans l'analyse de documents PDF.

Réponds de manière détaillée, claire et structurée.
Explique les informations importantes.
Quand c'est pertinent, fais des listes à puces.
Si on demande un résumé, fais un résumé complet.
Réponds uniquement à partir du contexte ci-dessous.
Si on demande des améliorations, donne des conseils précis.
Si l'information n'est pas dans le document, dis : "Je ne trouve pas cette information dans le document."

Contexte :
{context}

Question :
{query}
"""

# ─── Logique principale ──────────────────────────────────────────────────────
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if uploaded_files:
        current_hash = get_files_hash(uploaded_files)
        if st.session_state.get("current_hash") != current_hash:
            st.session_state.pop("collection", None)
            st.session_state.current_hash = current_hash
            st.info("🔄 Nouveaux fichiers détectés — retraitement en cours...")

        if "collection" not in st.session_state:
            collection_name = f"pdf_{current_hash}"
            existing = [c.name for c in chroma_client.list_collections()]

            if collection_name in existing:
                st.info("📂 PDFs déjà traités — chargement depuis la base existante")
                st.session_state.collection = chroma_client.get_collection(name=collection_name)
            else:
                with st.spinner("⏳ Traitement des PDFs en cours..."):
                    all_texts = []
                    for file in uploaded_files:
                        text = extract_text_from_pdf(file)
                        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
                        all_texts.extend(chunks)

                    embeddings = embed_texts(all_texts)
                    st.session_state.collection = create_chromadb_collection(
                        embeddings, all_texts, collection_name
                    )
                st.success(f"✅ {len(uploaded_files)} PDF(s) traités et sauvegardés !")

        collection = st.session_state.collection
        results = query_chromadb_collection(collection, query, n_results)
        prompt = build_prompt(results, query)
        response = query_ollama(prompt)
        response_text = response["message"]["content"]

        with st.chat_message("assistant"):
            st.markdown(response_text)
            with st.expander("📎 Voir les sources utilisées"):
                for i, doc in enumerate(results["documents"][0]):
                    st.caption(f"**Extrait {i+1} :**")
                    st.info(doc[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": response_text})

    else:
        st.warning("Veuillez uploader au moins un PDF")