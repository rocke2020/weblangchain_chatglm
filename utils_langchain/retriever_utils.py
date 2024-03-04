import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import bs4
import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores.chroma import Chroma
from pandas import DataFrame
from tqdm import tqdm
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

sys.path.append(os.path.abspath("."))


logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, datefmt='%y-%m-%d %H:%M',
    format='%(asctime)s %(filename)s %(lineno)d: %(message)s')


SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def does_vectorstore_exist(persist_directory: str, embeddings) -> bool:
    """
    Checks if vectorstore exists
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not db.get()["documents"]:
        return False
    logger.info("Found vectorstore at %s", persist_directory)
    return True


def get_hf_embedding_func(embedding_model="sentence-transformers-mini"):
    """
    seq length
        sentence-transformers-mini: 256 tokens/words.
        ll-mpnet-base-v2: 384 tokens
    """
    if embedding_model == "sentence-transformers-mini":
        model_path = "/mnt/nas1/models/sentence-transformers/all-MiniLM-L6-v2"
    else:
        model_path = "/mnt/nas1/models/sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    return embeddings


def get_embedding_func(embedding_model="bge-m3"):
    """
    embedding_model:
        bge-m3, dim 1024, seq length 8192, multilingua
        sentence-transformers/all-mpnet-base-v2, dim 768, seq length 384, English
    """
    if embedding_model == "bge-m3":
        model_path = "/mnt/nas1/models/BAAI/bge-m3"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif embedding_model.startswith("sentence-transformers"):
        embeddings = get_hf_embedding_func(embedding_model)
    else:
        raise ValueError("Invalid embeding_model")
    return embeddings


def get_docs(data_source="lilianweng-web"):
    if data_source == "lilianweng-web":
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
    # ic(docs)
    return docs


def get_retriever(
    docs,
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="bge-m3",
    persist_dir="/mnt/nas1/models/chroma/rag_2_lilianweng_web",
    overwrite_created_vectorstore=0,
):
    """Including embeddings, text split, create from split documents"""
    embeddings = get_embedding_func(embedding_model)
    if not overwrite_created_vectorstore and does_vectorstore_exist(persist_dir, embeddings):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(docs)
        ic(len(splits), splits[0].page_content[:100], splits[-1].page_content[:100])
        logger.info("Create new vectorstore and remove existent persist_dir")
        if Path(persist_dir).exists():
            logger.info("vectorstore_exist exists, remove it")
            shutil.rmtree(persist_dir)
            time.sleep(2)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=persist_dir
        )
    retriever = vectorstore.as_retriever()
    return retriever


if __name__ == "__main__":
    get_retriever(get_docs("lilianweng-web"))
