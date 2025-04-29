# 🏗️ Structural Code Chatbot for Civil Engineers

An interactive Streamlit-based chatbot that allows civil/structural engineers to chat with design code documents (e.g., ACI 318, Eurocode) for quick clause reference, compliance checks, and structural calculations.

---

## 🚀 Features

### 📥 Document Ingestion
- Upload **PDF**, **DOCX**, or **TXT** files of structural code documents
- Auto-loads local code files from a `data/` folder based on **keyword matching**

### 🔍 Retrieval-Augmented Generation (RAG)
- Uses **LangChain**, **FAISS**, and **MiniLM embeddings** to index and search document content
- Retrieves relevant clauses and commentary based on your query

### 💬 Conversational Chatbot
- Ask natural language questions like:
  - *“Which clause in ACI-318 covers beam design?”*
  - *“Is this design compliant with any code?”*
  - *“Calculate the moment capacity of beam B201 using b=300, d=500, fc=30, As=1500”*

### 🧮 Structural Calculator Agent
- Calculates **ϕMn (design moment capacity)** based on user input using:
  \[
  \phi M_n = \phi \cdot A_s \cdot f_y \cdot \left(d - \frac{a}{2}\right)
  \]
  Where:  
  - `b`: width of beam  
  - `d`: effective depth  
  - `fc`: concrete compressive strength  
  - `As`: steel area

### 🌍 Web Search Agent
- Automatically queries the **web** for structural engineering content if local files are insufficient
- Powered by **SerpAPI** and integrates seamlessly into the agent loop


---

## 🧠 Built With

- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI GPT](https://platform.openai.com/)
- [SerpAPI](https://serpapi.com/)
- [Streamlit](https://streamlit.io/)

---

## 📬 Future Ideas

- Support for **Revit/CAD integration**
- Upload and extract from **engineering drawings**
- Multi-language clause retrieval (e.g. Arabic + English)

---

