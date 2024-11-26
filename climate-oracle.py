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
# Ensure that OPENAI_API_KEY and ANTHROPIC_API_KEY are set up as environment variables with valid keys.
# Then install the Python requirements:
# pip install -r requirements.txt
#
# Implementation:
# --------------
# CLI leverages the code built in the accompanying notebook.
#
# TODO:
# -----
# 1. Add CLI input parameter support/ability to change
#
# History:
# -------
# 26.11.24    v0.1    First cut based on accompanying notebook
# 26.11.24    v0.2    Added Anthropic and Ollama LLMs plus CLI support
#

import os
import time
import logging
from logging import Logger
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import Runnable
from langchain_core.language_models import LLM
from typing import List, Dict

PROGRAM = __file__.split("/")[-1]
VERSION = "0.2"
DATE = "26.11.24"
AUTHOR = "<mal@malm.co.uk>"

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

def initLogger(verbose: bool) -> Logger:
    ''' Initialise standard Python console logging. '''
    if verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s"
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Suppress pypdf 'Ignoring wrong pointing object' warnings
        pypdf_logger = logging.getLogger("pypdf")
        pypdf_logger.setLevel(logging.WARNING)
    else:
        # this will silence all logging including from modules
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.FileHandler(os.devnull))
        # Suppress pypdf 'Ignoring wrong pointing object' warnings
        pypdf_logger = logging.getLogger("pypdf")
        pypdf_logger.setLevel(logging.ERROR)
    return logger

def getModel(model: str) -> LLM:
    if model == 'ollama':
        # See: https://python.langchain.com/docs/integrations/providers/ollama/
        # pip install -U langchain-ollama
        name = 'llama3.2'
        print(f"using '{name}'")
        llm = OllamaLLM(model=name, temperature=0)
    elif model == 'claude':
        # https://python.langchain.com/docs/integrations/providers/anthropic/
        # pip install -U langchain-anthropic
        # Model to use is 'claude-3-5-sonnet-latest'
        name = 'claude-3-5-sonnet-latest'
        print(f"using '{name}'")
        llm = ChatAnthropic(model=name, temperature=0)
    else:
        # Default to OpenAI gpt-3.5
        # See: https://python.langchain.com/docs/integrations/providers/openai
        name = 'gpt-3.5-turbo-instruct'
        print(f"using '{name}'")
        llm = OpenAI(model=name, temperature=0)
    return llm

def generateChain(pdfs: List, model: str) -> Runnable:
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
    llm = getModel(model)
        
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
    
def main(arguments: Dict):
    verbose = False
    logger = initLogger(False)
    model = "openai"
    if arguments.get("--verbose"):
        verbose = True
        logger = initLogger(verbose)
        logger.info(f"::main() - arguments =\n{arguments}")
    if arguments.get("--model"):
        model = str(arguments.get("--model")[0])

    if arguments.get("--version") or arguments.get("-V"):
        print(f"{PROGRAM} version {VERSION}.  Released {DATE} by {AUTHOR}")
    elif arguments.get("--help") or arguments.get("-h"):
        print(usage)
    else:
        t0 = time.time()
        pdfs = ["https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf"]
        print(f"Generating create_retrieval_chain using llm='{model}'...") 
        climate_qa_chain = generateChain(pdfs, model)        
        t1 = time.time()
        print(f"successfully built RAG in {round((t1-t0),2)} seconds and will use {model} LLM")
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


if __name__ == "__main__":
    import docopt

    usage = """
    {}
    ------------------
    Usage:
    {} [-m <model>] [-v]
    {} -h | --help
    {} -V | --version
    Options:
    -h, --help                          Show this screen.
    -v, --verbose                       Verbose mode.
    -V, --version                       Show version.
    -m <model>, --model <model>         LLM model.
    Examples
    1. Generate climate oracle using Anthropic Claude LLM:
    {} -m claude -v
    """.format(
        *tuple([PROGRAM] * 5)
    )
    arguments = docopt.docopt(usage)
    main(arguments)
