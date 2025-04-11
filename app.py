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

import torch
torch.classes.__path__ = [] # add this line to manually set it to empty. 


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
           
st.set_page_config(page_title="Smart Data Insight", layout="wide")
st.title("Smart Data Insights: AI-Powered, Time-Saving, Productivity-Boosting - Demo Version")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": False}
# )

# vectorstore = Chroma(
#     collection_name="financial_analysis",
#     embedding_function=embeddings,
#     persist_directory="./openai_vector_db"
# )
vectorstore = FAISS(collection_name="financial_analysis",
    embedding_function=embeddings,
    persist_directory="./openai_vector_db")

financial_prompt = """
You are a financial analyst providing detailed insights based on the provided data.
- Answer the user's question based on the text, tables, and images extracted from the document.
- Response should be structured, concise, and in markdown format
Always include the reasoning behind your answers.
"""

def process_pdf_to_vectorstore(file_path):
    chunks = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Table", "Image"],  # ✅ Extract only structured blocks
        extract_image_block_to_payload=False,          # ✅ Do not embed base64 into chunks
        infer_table_structure=False,                   # ✅ Avoid layout model for tables
        ocr_languages=None,                            # ✅ Skip OCR
        extract_from_layout=False,                     # ✅ <- this disables layout OCR agent
        hi_res_model_name=None,                       # ✅ prevents detectron2 layout model load
        pdf_infer_table_structure=False,              # ✅ double-disable internal fallback
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

    # images = get_images_base64(chunks)

    text_embeddings = embeddings.embed_documents(texts)
    for chunk, embedding in zip(texts, text_embeddings):
        vectorstore.add_documents([Document(page_content=chunk, metadata={})], embeddings=[embedding])

    table_chunks = [table.text for table in tables]
    table_embeddings = embeddings.embed_documents(table_chunks)
    for chunk, embedding in zip(table_chunks, table_embeddings):
        vectorstore.add_documents([Document(page_content=chunk, metadata={})], embeddings=[embedding])

    # for image in images:
    #     vectorstore.add_documents([Document(page_content="Image content", metadata={"image_base64": image})])

    vectorstore.persist()

def parse_docs(docs):
    images_b64 = []
    text_contents = []
    for doc in docs:
        if "image_base64" in doc.metadata:
            images_b64.append(doc.metadata["image_base64"])
        else:
            text_contents.append(doc.page_content)
    return {"texts": text_contents, "images": images_b64}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    context_text = "\n".join(docs_by_type["texts"])
    prompt_template = f"""
    {financial_prompt}

    Context:
    {context_text}

    Question: {kwargs["question"]}
    """
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_template)])

def retrieve_from_vectorstore(query):
    return vectorstore.similarity_search(query, k=10)

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

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("./uploads", uploaded_file.name)
    os.makedirs("./uploads", exist_ok=True)

    if "processed" not in st.session_state:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Processing PDF.."):
            process_pdf_to_vectorstore(file_path)
            st.session_state["processed"] = True
            st.success("PDF processed successfully!")
    else:
        st.info("✅ PDF already processed.")

    user_query = st.text_input("Ask your question:", placeholder="e.g. Who is the Director & COO?")
    if user_query:
        with st.spinner("Getting answer from model..."):
            response = query_chain(user_query)
            #st.markdown("### 📌 Response")
            #st.markdown(f"```{response}```")
            st.markdown(response, unsafe_allow_html=True)
