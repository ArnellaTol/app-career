import streamlit as st
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import numpy as np

@st.cache_data(show_spinner="Loading JSONL files...")
def load_jsonl_files(folder_path):
    all_records = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except json.JSONDecodeError:
                        print(f"Error decoding line in file: {filename}")
    return all_records

rag_data = load_jsonl_files("./jsonl datafiles")


@st.cache_resource(show_spinner="Logging into Hugging Face...")
def login_hf():
    os.environ["HF_TOKEN"] = "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"
    login(token=os.environ["HF_TOKEN"])

login_hf()


@st.cache_resource(show_spinner="Loading LLaMA-3.2 model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    return tokenizer, model

tokenizer, model = load_model()


@st.cache_resource
def build_annoy_index():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [item["text"] for item in rag_data]
    sources = [item["source"] for item in rag_data]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    annoy_index = AnnoyIndex(dimension, 'angular')

    for i, vec in enumerate(embeddings):
        annoy_index.add_item(i, vec)

    annoy_index.build(10)
    
    return embedder, annoy_index, texts

embedder, annoy_index, texts = build_annoy_index()



def generate_career_advice(question: str, tokenizer, model) -> str:
    prompt = prompt = f"""
You are a career advisor for high school students.
Your only task is to select 3 to 5 career paths that are the best possible match for the student's stated interests, strengths, and dislikes.

Student’s message:
{question}

School Career Advisor’s answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip().split("\n")[0]


def generate_rag_career_advice(question: str, tokenizer, model, embedder, annoy_index, texts: list, k: int = 5) -> str:
    
    query_embedding = embedder.encode([question], convert_to_numpy=True)

    indices = annoy_index.get_nns_by_vector(query_embedding[0], k, include_distances=False)
    context_docs = [texts[i] for i in indices]

    context = "\n\n".join(context_docs)
    prompt = f"""
You are a career advisor for high school students.

You have access to relevant background knowledge about career paths, student preferences, and educational strategies, shown below.

Context:
{context}

Your only task is to select NO MORE THAN ONE or maximum TWO career paths that are the best possible match for the student's stated interests, strengths, and dislikes.
DO NOT GENERATE CONTINUING DIALOGUE, ANSWER TO STUDENT'S QUESTION ONLY ONCE.
Strict instructions:
- Base your suggestions strictly on the student’s message. Do not invent or assume anything not mentioned.
- Recommend only career paths that clearly align with what the student enjoys and is good at, and that avoid what they dislike or find difficult.
- For each suggested path, explain in 1–2 sentences why it fits this student specifically.
- Do not give general advice or list unrelated options "just in case."
- Do not exceed 2 suggestions. Do not use bullet points or numbered lists.
- Keep the total response under 100 words. Be focused and relevant.

Student’s message:
{question}

Career Advisor’s answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=80,
            do_sample=True,
            top_p=0.9,
            temperature=0.3,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

        if not answer.endswith("."):
            last_period = answer.rfind(".")
            if last_period != -1:
                answer = answer[:last_period + 1]
            else:
                answer = answer.strip()

        return answer



st.title("AI Career Advisor for High School Students")

student_question = st.text_area("Enter your question about careers:", height=100)

use_rag = st.toggle("Enable RAG (Retrieval-Augmented Generation)")

if st.button("Get Advice"):
    with st.spinner("Generating responses..."):
        if use_rag:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("💡 Base Model")
                base_answer = generate_career_advice(student_question, tokenizer, model)
                st.write(base_answer)

            with col2:
                st.subheader("📚 RAG-enhanced Model")
                rag_answer = generate_rag_career_advice(student_question, tokenizer, model, embedder, annoy_index, texts)
                st.write(rag_answer)

        else:
            st.subheader("AI Career Advice")
            base_answer = generate_career_advice(student_question, tokenizer, model)
            st.write(base_answer)


        
