from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community import embeddings
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


def loadCSV(df):
    # This function creates document embeddings and returns a chromadb vector store as output. Currently this only works with test.csv file in the repo but
    # changing columns_to_embed and columns_to_metadata to match other types as wll. If the context is longer than 1000 tokens then increase those as well.
    
    columns_to_embed = ["Number", " Incident Description", " Notes"]
    columns_to_metadata = [
        "Number",
        " Incident State",
        " Active",
        " Reassignment Count",
        " Reopen Count",
        " Sys Mod Count",
        " Made SLA",
        " Caller ID",
        " Opened By",
        " Opened At",
        " Sys Created By",
        " Sys Created At",
        " Sys Updated By",
        " Sys Updated At",
        " Contact Type",
        " Location",
        " Category",
        " Subcategory",
        " U Symptom",
        " CMDB CI",
        " Impact",
        " Urgency",
        " Priority",
        " Assignment Group",
        " Assigned To",
        " Knowledge",
        " U Priority Confirmation",
        " Notify",
        " Problem ID",
        " RFC",
        " Vendor",
        " Caused By",
        " Closed Code",
        " Resolved By",
        " Resolved At",
    ]


    docs = []
    for index, row in df.iterrows():
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        to_embed = "\n".join(
            f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items()
        )
        newDoc = Document(page_content=to_embed, metadata=to_metadata)
        docs.append(newDoc)

    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=0, length_function=len
    )

    all_splits = splitter.split_documents(docs)
    embedding = embeddings.OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)
    
    return vectorstore