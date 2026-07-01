# IntelFetchRAG 🕵️‍♂️📰

IntelFetchRAG is a privacy-first, fully local Retrieval-Augmented Generation (RAG) research assistant built with Streamlit and LangChain. It allows analysts and researchers to scrape data from multiple news URLs securely, segment the text, and query an isolated local vector store using open-source Large Language Models (LLMs)—**ensuring no data ever leaves your computer.**

---

## 🎯 Engineering Highlights & Architecture Choices

When designing this system, production efficiency, cost-optimization, and data privacy were treated as first-class constraints. 

### 1. Zero-Egress Data Isolation (100% On-Device)
* **Problem:** Traditional RAG stacks rely on cloud APIs (like OpenAI or Pinecone), which introduce corporate compliance risks, vendor lock-in, and unpredictable data egress fees.
* **Solution:** Orchestrated an end-to-end local framework using `Ollama` (`llama3.2:1b`) and `FAISS`. Enterprise data remains entirely bound to local volatile memory and disk infrastructure.

### 2. Dual-Layer Caching Architecture
To guarantee ultra-low latency execution and minimize compute/network degradation, the application implements an aggressive caching strategy:
* **Resource Thread Caching (`@st.cache_resource`):** Restricts the heavy model weight loading tensors (`ChatOllama` and `OllamaEmbeddings`) to a single execution on startup. Subsequent interactions yield a `O(1)` memory access time.
* **Network Data Caching (`@st.cache_data`):** Intercepts raw data ingestion. If an analyst processes an unchanged list of URLs, the system bypasses the network completely, eliminating duplicate web requests and protecting system scaling boundaries.

### 3. Production-Grade Streamlit Lifecycle Management
* Implemented stateful `st.empty()` viewports to handle asynchronous loading updates natively.
* Configured programmatic User-Agent request headers to prevent service blocking during heavy scraping procedures.

---

## 🛠️ System Architecture Diagram

```text
               [Analysts Web URLs Ingestion]
                            │
                            ▼ (Custom Header / Compliant User-Agent)
                 [UnstructuredURLLoader]
                            │
                            ▼
               [RecursiveCharacterTextSplitter] 
              (Chunk Size: 1000, Overlap: 100)
                            │
                            ▼
          [OllamaEmbeddings: nomic-embed-text] ──► [FAISS Local Store (Disk)]
                                                             │
                                                             ▼ (Retriever k=3)
[System Prompt Optimization] ──► [LCEL Pipeline Engine] ◄────┘
                                         │
                                         ▼ (LLM: Llama 3.2 1b)
                            [Synthesized Answer Output]



```
---

## ⚙️ Local Setup & Installation

Make sure that you clone the repositery
### 1. Engine Installation
Ensure you have **Ollama** active on your system. If not installed, download the client from [ollama.com](https://ollama.com).
Open your `System terminal` and pull the exact embedding and inference models required by this project:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text:latest
```

### 2.Install Requirements

Inside the `terminal of your code editor` type these commands
```bash
pip install -r requirements.txt
```

### 3.Execution

Activate the streamlit app
```bash
streamlit run code.py
```
