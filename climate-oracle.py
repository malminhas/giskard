#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# climate-oracle.py
# -----------------
# (c) 2024 Mal Minhas, <mal@malm.co.uk>
#
# RAG climate question and answer CLI built on IPCC AR6 Synthesis Report.
#
# Installation:
# -------------
# pip install -r requirements.txt
#
# Implementation:
# --------------
# CLI leverages the code built in the accompanying notebook.
#
# History:
# -------
# 26.11.24    v0.1    First cut based on accompanying notebook
#

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import Runnable
from typing import List, Dict

# Suppress pypdf 'Ignoring wrong pointing object' warnings
import logging
logger = logging.getLogger("pypdf")
logger.setLevel(logging.ERROR)

system_prompt = """You are the Climate Assistant, a helpful AI assistant.
Your task is to answer common questions on climate change.
You will be given a question and relevant excerpts from the IPCC Climate Change Synthesis Report (2023).
Please provide comprehensive answers based on the provided context.  Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

def generateChain(pdfs: List) -> Runnable:
    # Prepare vector store (FAISS) with IPPC report(s).  Store splits in vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    for i, pdf in enumerate(pdfs):
        loader = PyPDFLoader(pdf)
        if i == 0:
            vectorstore = FAISS.from_documents(documents=loader.load_and_split(text_splitter), embedding=OpenAIEmbeddings())
        else:
            vectorstore_i = FAISS.from_documents(documents=loader.load_and_split(text_splitter), embedding=OpenAIEmbeddings())
            vectorstore.merge_from(vectorstore_i)
    vectorstore.save_local('.')
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": len(pdfs)})
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    
    # Question-answering against an index using create_retrieval_chain:    
    prompt = PromptTemplate(template=system_prompt, input_variables=["question", "context"])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
    return qa_chain

def getAnswer(qa_chain: Runnable, question: str) -> Dict:
    r = {}
    with get_openai_callback() as cb:
        answer = qa_chain.invoke({"input": system_prompt,"question": question})
        r['answer'] = answer.get('answer').strip()
        references = answer.get('context')
        r['references'] = []
        for i,reference in enumerate(references):
            d = reference.to_json()
            source = d.get('kwargs').get('metadata').get('source')
            page = d.get('kwargs').get('metadata').get('page')
            contents = d.get('kwargs').get('page_content')
            r['references'].append(f"{source},{page}")
        r['cost'] = round(cb.total_cost,4)
        r['prompt_tokens'] = cb.prompt_tokens
        r['completion_tokens'] = cb.completion_tokens
        r['total_tokens'] = cb.total_tokens
    return r
    
if __name__ == '__main__':
    pdfs = ["https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf"]
    print(f"Generating create_retrieval_chain...") 
    climate_qa_chain = generateChain(pdfs)
    print("Enter a climate question (or press Enter to quit). eg. 'Is sea level rise avoidable and when will it stop?'")
    while True:
        try:
            # Prompt the user to enter a sentence
            question = input("> ")
            # Break the loop if the user enters an empty string
            if question == "":
                print("Empty input.  Exiting the program. Goodbye!")
                break
            answer = getAnswer(climate_qa_chain, question)
            print(f"{answer.get('answer')}")
        except:                
            print("Interrupt.  Exiting the program. Goodbye!")
            break
     
