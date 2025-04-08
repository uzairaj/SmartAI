import streamlit as st
from unstructured.partition.pdf import partition_pdf
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
import pdfplumber
import base64
import os
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
           
st.set_page_config(page_title="Smart Data Insight", layout="wide")
st.title("Smart Data Insights: AI-Powered, Time-Saving, Productivity-Boosting - Demo Version")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

embeddings = OpenAIEmbeddings()

faiss_index_path = "./faiss_index"
vectorstore = None

# Load existing FAISS index if available
if os.path.exists(faiss_index_path):
    try:
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning("⚠️ Failed to load existing FAISS index: " + str(e))


def process_pdf_to_vectorstore(file_path):
    global vectorstore
    chunks = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Table", "Image"],  # ✅ Still get tables/images
        extract_image_block_to_payload=False,
        split_pdf_page=True,
        max_characters=10000
    )

    texts = []
    tables = []
    images = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

    # def get_images_base64(chunks):
    #     images_b64 = []
    #     for chunk in chunks:
    #         if "CompositeElement" in str(type(chunk)):
    #             for el in chunk.metadata.orig_elements:
    #                 if "Image" in str(type(el)):
    #                     images_b64.append(el.metadata.image_base64)
    #     return images_b64

    #images = get_images_base64(chunks)

    all_docs = [Document(page_content=t) for t in texts]
    all_docs.extend([Document(page_content=tbl.text) for tbl in tables])
    #all_docs.extend([Document(page_content="Image content", metadata={"image_base64": img}) for img in images])

    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(faiss_index_path)


def parse_docs(docs):
    images_b64 = []
    text_contents = []
    for doc in docs:
        #lse:
        text_contents.append(doc.page_content)
    return {"texts": text_contents}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    context_text = "\n".join(docs_by_type["texts"])
    prompt_template = f"""
    You are a financial analyst providing detailed insights based on the provided data.
    - Answer the user's question based on the text, tables, and images extracted from the document.
    - Response should be structured, concise, and in json format
    - If the user asks for data in images, describe the image content based on the extracted context.
    Always include the reasoning behind your answers.

    Context:
    {context_text}

    Question: {kwargs["question"]}
    """
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_template)])

def retrieve_from_vectorstore(query):
    global vectorstore
    return vectorstore.similarity_search(query, k=10) if vectorstore else []

def query_chain(user_question):
    vectorstore_runnable = RunnableLambda(retrieve_from_vectorstore)
    chain = (
        {
            "context": vectorstore_runnable | RunnableLambda(parse_docs),
            "question": RunnableLambda(lambda x: x),
        }
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )
    return chain.invoke(user_question)

uploaded_file = st.file_uploader("Upload your financial PDF", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("./uploads", uploaded_file.name)
    os.makedirs("./uploads", exist_ok=True)

    if "processed" not in st.session_state:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Processing PDF and building vectorstore..."):
            process_pdf_to_vectorstore(file_path)
            st.session_state["processed"] = True
            st.success("PDF processed successfully!")
    else:
        st.info("✅ PDF already processed.")

    user_query = st.text_input("Ask your question:", placeholder="e.g. Who is the Director & COO?")
    if user_query:
        with st.spinner("Getting answer from model..."):
            response = query_chain(user_query)
            st.markdown("### 📌 Response")
            st.markdown(response, unsafe_allow_html=True)