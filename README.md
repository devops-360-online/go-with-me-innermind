# Building an LLM‐based Collaboration System: A Beginner's Roadmap

This guide explains key concepts (MCP, Agent2Agent/A2A, ADK, RAG) and tools (LangChain, Ray, MLflow) in simple terms, then shows step-by-step how to build an open‐source LLM system on GCP/Kubernetes. The goal is an enterprise "inner source" assistant that helps teams collaborate, discover projects and tools, and access documentation. We'll use Python (for most AI code) and Go (for microservices/tools) on Google Kubernetes Engine, and open‐source models like Mistral or LLaMA.

## Key Concepts & Tools

### Model Context Protocol (MCP)

MCP is an open protocol for feeding external data or tools into an LLM at inference time [openai.github.io](https://openai.github.io/openai-agents-python/mcp/). In simple terms, MCP is like a USB-C port for AI: it lets your LLM connect to standardized "tool servers" (e.g. a database, file system, or search API) to fetch relevant context. For example, an LLM agent can use an MCP file-system server that exposes your project docs. Using an Agents SDK (like Google's ADK), you can attach MCP servers so the agent calls them as needed. The MCP spec defines two server types – local (stdio subprocesses) or remote (HTTP/SSE) – which your code can invoke (e.g. using MCPServerStdio to run a filesystem tool) [openai.github.io](https://openai.github.io/openai-agents-python/mcp/). This way, when the LLM needs extra info (e.g. code snippets or docs), it calls the MCP server rather than hallucinating.

### Agent2Agent (A2A) Protocol

Agent2Agent (A2A) is Google's new open standard for letting multiple AI agents communicate and collaborate [developers.googleblog.com](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) [github.com](https://github.com/google/A2A). Think of it as a common language for agents: each agent publishes an "Agent Card" (a JSON at /.well-known/agent.json) describing its skills and endpoint, and exposes an HTTP API for tasks. For example, an A2A server is an agent that can receive tasks via a REST API, and an A2A client is any code or agent that calls that API to assign work. A2A tasks go through states (submitted, working, completed, etc.), and can use server‐sent events (SSE) for streaming updates. In practice, A2A lets a coding assistant agent talk to a documentation agent, or a support agent ask a finance agent for data. As Google notes, "A2A is an open protocol that provides a standard way for agents to collaborate with each other, regardless of the underlying framework or vendor" [developers.googleblog.com](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/). This is still very new, but it means your teams' agents (even from different tools) can interoperate on the same network.

### Google Agent Development Kit (ADK)

The Agent Development Kit (ADK) is Google's open‐source Python framework for building and deploying AI agents [cloud.google.com](https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-multi-system-agents-with-vertex-ai). It abstracts away much of the plumbing so you can define agents, tools, workflows and multi-agent systems in Python code. In ADK you create an Agent by giving it a name, a prompt model, instructions, and a list of tool functions. For example, you might write Python functions that search your codebase or fetch wiki docs, and pass them as `tools=[search_code, lookup_doc]` when instantiating `Agent(...)`. ADK then automatically converts those functions into callable tools for the LLM to use. Importantly, ADK is model- and deployment-agnostic [google.github.io](https://google.github.io/adk-docs) [cloud.google.com](https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-multi-system-agents-with-vertex-ai): you choose any LLM (Gemini or open models) and deploy on any platform (local, Cloud Run, Kubernetes). ADK even supports MCP for secure data integration [cloud.google.com](https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-multi-system-agents-with-vertex-ai). In short, ADK makes agent development "feel more like software development" [google.github.io](https://google.github.io/adk-docs): you write Python code for agents and tools, test locally, and then deploy containers or use Vertex AI's Agent Engine for production.

### Retrieval-Augmented Generation (RAG)

RAG is a technique that combines an LLM with an external knowledge source to improve answers [python.langchain.com](https://python.langchain.com/docs/concepts/rag/). Here's how it works: when the user asks a question, the system retrieves (searches) relevant documents or data from a database (e.g. your internal wiki, code repo, support tickets). Then it appends that retrieved content to the LLM's prompt so the model can ground its answer in real information. This solves the LLM's "knowledge cutoff" problem by giving it up-to-date facts. For example, if someone asks about the company's latest API, the RAG system would first pull the API spec from documentation, then feed it into the LLM to generate a precise answer. The diagram below illustrates RAG: 

[https://python.langchain.com/docs/concepts/rag/](https://python.langchain.com/docs/concepts/rag/)

Figure: RAG process – (1) a Retrieval System searches a knowledge base for relevant documents, then (2) a Language Model (LLM) is prompted with those documents to generate an answer [python.langchain.com](https://python.langchain.com/docs/concepts/rag/). RAG is key for our system: by indexing all project wikis, code docs, and files into a searchable database (vector store), the assistant always answers using company knowledge. This makes answers accurate ("based on our docs") and keeps internal practices consistent.

### LangChain, Ray, and MLflow

#### LangChain
An open-source Python framework for building LLM applications. LangChain provides chains and agents that connect LLM calls with tools, memories, and external data. In simple terms, LangChain lets you "chain together" LLM prompts, calls to APIs, and logic flows. For example, LangChain offers built-in support for RAG: you can create a RetrievalQA chain that takes a question, uses a vectorstore to retrieve docs, and then calls your LLM with the combined prompt. As LangChain's docs say, it is a "composable framework to build with LLMs" [langchain.com](https://www.langchain.com/). Beginners often use it to rapidly prototype chatbots or assistants that call APIs, store chat history, or perform multi-step reasoning.

#### Ray
An open-source compute framework for scaling Python workloads, especially ML tasks. Ray lets you distribute and parallelize any Python function across a cluster. For instance, you might use Ray to index millions of documents into your vector store in parallel, or to serve your LLM model at scale. Ray includes libraries like Ray Serve (for model serving), RLlib (RL training), and Tune (hyperparameter tuning). As Google explains, "Ray is an open-source framework for scaling AI and Python applications" and provides the infrastructure for distributed computing [cloud.google.com](https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/overview). In practice, you could use a Ray cluster on GKE to run your vector DB and inference nodes, ensuring high throughput.

#### MLflow
An open-source platform for managing the ML lifecycle. MLflow helps you log experiments, version models, and deploy them. For example, when you fine-tune or benchmark different LLMs on your data, MLflow can log each run's parameters (model type, prompt variants) and metrics (accuracy, latency). It also provides a Model Registry to store and promote models (e.g. "select best open-model for code search"). As the docs state, "MLflow is an open-source platform… to assist ML teams… ensuring each phase is manageable, traceable, and reproducible" [mlflow.org](https://mlflow.org/docs/2.8.0/). In our system, use MLflow to track LLM evaluation on company benchmarks (like question-answer accuracy) and to manage which model is in production.

## Step-by-Step Implementation

Below is a practical roadmap. We assume no prior LLMOps knowledge. Each Step builds on the previous, from setup to deployment.

### 1. Prerequisites & Setup

- **Kubernetes Cluster**: Ensure you have a running K8s cluster and kubectl configured.
- **MLflow Tracking Server**: A deployed MLflow server (URI) to log experiments.
- **Language Models**: We will use an open-source LLM (e.g. Mistral 7B-Instruct or LLaMA2-7B). Ensure you have access (via HuggingFace Hub or local weights).
- **Python Environment**: Install necessary packages in a virtualenv: Ray, LangChain, Transformers, FAISS/Qdrant client, MLflow, and ADK. For example:

```bash
pip install ray[default] langchain transformers sentence-transformers faiss-cpu mlflow google-adk fastapi uvicorn pydantic
```

- **MCP/ADK Tools**: We will use Google's ADK (Agent Development Kit) for agents (pip install google-adk). We also use fastmcp or similar to expose MCP endpoints.
- Finally, plan your architecture: usually, you'll have a vector DB (e.g. a Qdrant or FAISS index) as a service, agents as microservices, and possibly a Ray cluster for heavy computation. You may use Helm charts or Docker/K8s YAML to deploy these components.

### Step 2: Deploy an Open-Source LLM

1. **Choose a model**: Pick a high-quality open model like Mistral-7B or Meta's Llama-2. For example, Mistral-7B Chat can be fine for English text. Download the model weights from Hugging Face or the provider.

2. **Containerize the model**: Create a Docker image that loads the model and exposes an HTTP/gRPC endpoint. You can use libraries like vLLM or text-generation-webui inside the container to serve the model. For instance, a simple Flask app could load Llama2 and respond to /generate requests.

3. **Deploy on K8s**: Define a Kubernetes Deployment (and a Service) for the model server. Example (YAML simplified):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: llm-serving }
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: llm-server
        image: gcr.io/my-project/llm-mistral:latest
        resources:
          limits: { nvidia.com/gpu: 1 }   # Use GPU
```

4. **Testing**: Verify you can hit the model endpoint. For example, in Python:

```python
import requests
resp = requests.post("http://llm-serving.default.svc.cluster.local/generate",
                    json={"prompt": "Hello, world!"})
print(resp.json())
```

Replace with your actual service address. Ensure latency is acceptable; autoscale with an HPA if needed.

### Step 3: Prepare the Knowledge Base (Innersource Data)

### 2. Ingest Internal Documents and Build Vector DB

First, gather your internal documents (docs, FAQs, code snippets, etc.) and embed them into a vector database for retrieval. Ray helps scale this ingestion. We'll use Ray to parallelize embedding and Qdrant (or FAISS) to store vectors for fast semantic search [medium.com](medium.com) [python.langchain.com](python.langchain.com).

**Convert Documents to Text**: Load documents from your file system or repository. For example, using Python:

```python
import glob
from pathlib import Path

doc_paths = glob.glob("/mnt/docs/**/*.txt", recursive=True)  # adjust path/pattern
documents = []
for path in doc_paths:
    with open(path, "r", encoding="utf-8") as f:
        documents.append(f.read())
```

**Initialize Ray**: Launch Ray on your cluster (or locally) for parallel tasks.

```python
import ray
ray.init(address="auto")  # or ray.init() for local, or use ray cluster config
```

**Define an Embedding Worker**: Use a lightweight embedding model (e.g. SentenceTransformer or FastEmbed). Wrap it as a Ray remote actor to enable parallel embedding [medium.com](medium.com).

```python
from sentence_transformers import SentenceTransformer

@ray.remote
class EmbedWorker:
    def __init__(self):
        # Use a sentence-transformer model for embeddings
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

# Example: create 4 workers
workers = [EmbedWorker.remote() for _ in range(4)]
```

**Chunk Documents and Parallel Embed**: Split your documents into chunks, send to workers, and collect embeddings.

```python
chunks = [documents[i::4] for i in range(4)]  # simple round-robin split
embed_tasks = [worker.embed_text.remote(chunk) for worker, chunk in zip(workers, chunks)]
all_embeddings = ray.get(embed_tasks)
# Flatten list of lists:
embeddings = [vec for sublist in all_embeddings for vec in sublist]
```

**Store in Vector DB**: Using Qdrant or FAISS, create a collection and upload embeddings. Example with Qdrant [medium.com](medium.com) [python.langchain.com](python.langchain.com):

```bash
# Deploy Qdrant on Kubernetes (or run via Docker)
# Example YAML (qdrant-deployment.yaml):
apiVersion: apps/v1
kind: Deployment
metadata: { name: qdrant }
spec:
  selector: { matchLabels: { app: qdrant } }
  template:
    metadata: { labels: { app: qdrant } }
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports: [{ containerPort: 6333 }]
---
apiVersion: v1
kind: Service
metadata: { name: qdrant-service }
spec:
  selector: { app: qdrant }
  ports: [{ port: 6333, targetPort: 6333 }]
```

Apply with kubectl apply -f qdrant-deployment.yaml. Then in Python, connect and upload:

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="qdrant-service:6333")  # or "localhost"
collection = "internal_docs"
client.recreate_collection(
    collection_name=collection,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
)
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"doc_path": doc_paths[i]})
    for i in range(len(embeddings))
]
client.upload_points(collection_name=collection, points=points)
print(f"Stored {len(embeddings)} embeddings in Qdrant.")
```

Citation: LangChain recommends Qdrant as a scalable vector store for semantic search [python.langchain.com](python.langchain.com).

**Verify Retrieval**: You can test a query against the vector DB using cosine similarity:

```python
query = "How to deploy our service on Kubernetes?"
query_emb = SentenceTransformer("all-MiniLM-L6-v2").encode([query]).tolist()
results = client.search(collection, query_vector=query_emb[0], limit=3)
for res in results:
    print(res.payload["doc_path"], "(score:", res.score, ")")
```

At this point you have a vector index of your documents for RAG. The next sections show how to build agents that query this index.

### Step 4: Implement Retrieval-Augmented Generation (RAG)

With the data prepared, build the RAG pipeline. We will use LangChain for simplicity:

1. **Setup LangChain chain**: Use a retrieval chain that fetches documents then calls the LLM. For example:

```python
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceLLM

# Load vector store (from Step 3)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = HuggingFaceLLM(model_name="mistral-7b", temperature=0.0)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

2. **Querying**: When a user asks a question, run the chain:

```python
answer = qa_chain({"query": "How to onboard new contributors?"})
print(answer["result"])
```

LangChain will (1) retrieve top-5 relevant docs about onboarding, then (2) prompt the LLM with those docs plus the question. This produces an answer grounded in your innersource docs.

3. **In practice**: Deploy this as a service (e.g. a FastAPI app) on K8s so your agents or UI can call it. If you're concerned about latency, use Ray Serve to host the chain with autoscaling. Ray can parallelize heavy workloads like embedding large queries or serving many requests.

### Step 5: Develop Agents with ADK

Use the Agent Development Kit to create one or more intelligent agents. For example, you might build an agent that handles developer queries, and another for project discovery.

1. **Install ADK**: In your Python environment: `pip install google-adk`.

2. **Define tool functions**: Write Python functions for the agent's tools. Examples: `search_codebase(query)`, `lookup_doc(query)`, `list_projects()`, etc. Each returns structured results (e.g. status + data).

3. **Create an Agent**: In code, tie it together. For instance:

```python
from google.adk.agents import Agent

# Example tool: search documentation vectorstore via RAG chain
def doc_search_tool(query: str) -> dict:
    answer = qa_chain({"query": query})
    return {"status": "success", "result": answer["result"]}

# Example tool: call GitHub API (written in Go, but here wrapped via tool)
def repo_search_tool(query: str) -> dict:
    # (In practice, call a microservice written in Go or Python)
    res = github_search_api(query)
    return {"status": "success", "results": res}

my_agent = Agent(
    name="team_assistant",
    model="mistral-7b-chat",
    description="Helps developers find code and docs across innersource projects.",
    instruction="You are a friendly assistant who uses internal tools to find information.",
    tools=[doc_search_tool, repo_search_tool]
)
```

4. **Run the agent**: The ADK SDK provides ways to run the agent locally or in a dev UI. You can feed it user messages, and it will autonomously call the appropriate tools (using your functions) as needed.

5. **State and Memory**: ADK supports sessions and memory; you can choose to remember past interactions or not, depending on your use case. For example, you could enable an in-memory or persistent session so the agent "remembers" ongoing tasks.

(See the earlier ADK Quickstart example [google.github.io](https://google.github.io/adk-docs/get-started/quickstart/) for a template: define tool functions, then pass them into Agent(...).)

### Step 6: Integrate Tools via MCP

To securely connect agents with your data sources, use MCP with your ADK agents:

1. **MCP Servers**: Deploy MCP tool servers for anything that requires external context. For example, you could run @modelcontextprotocol/server-filesystem as a subprocess or container to give file access. Or write a simple MCP HTTP server that wraps a proprietary search API.

2. **Connect to Agent**: In ADK, add MCP servers to your agent's runtime. Pseudocode:

```python
from openai_agents.mcp import MCPServerStdio

# Example: local filesystem MCP server (Node.js)
mcp_server = await MCPServerStdio(params={
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data/docs"]
})
my_agent.add_mcp_server(mcp_server)
```

When the agent runs, it will auto-call list_tools() on the MCP server so it knows what tools are available (e.g. "read file", "search folder"). If the LLM decides to use one, the SDK will call_tool() on the MCP server.

3. **Usage**: Now your agent can say, for example, "Use your file tool to open the architecture doc." and it will fetch that data from the MCP server. This tightly couples LLM reasoning with actual company data.

### Step 7: Enable Multi-Agent Collaboration (Agent2Agent)

If you want separate agents to work together (for example, one agent for frontend devs and another for backend devs), use the A2A protocol:

1. **Run A2A servers**: Each agent deployment (e.g. as a container) should host an HTTP endpoint implementing A2A (many frameworks will add this). Expose .well-known/agent.json describing its capabilities.

2. **Discovery & Tasks**: One agent (the client) can fetch another's Agent Card to learn how to contact it. Then it initiates a task via A2A's /tasks/send API with a message. For example, a "project lookup" agent could ask a "data team" agent: "find all docs about data lakes". The A2A server on the data agent would receive this task and process it, streaming back results.

3. **Example flow**: (Using the GitHub A2A library)
   - Agent A discovers Agent B's endpoint and reads its Agent Card (capabilities).
   - Agent A (A2A client) sends a JSON tasks/send request to Agent B's /tasks/send URL.
   - Agent B processes the task, possibly involving its own LLM or tools.
   - If it's long-running, Agent B streams progress via SSE to Agent A.

4. **Why A2A**: This lets teams run agents independently but still have them collaborate. For example, a security agent (on Go) and a dev agent (on Python) can interoperate. As Google notes, A2A gives a "common language" so agents can share info across ecosystems [github.com](https://github.com/google/A2A). Note: A2A is very new, so starting with one agent or simple HTTP calls may suffice initially.

### Step 8: Scale and Orchestrate with Ray and Kubernetes

1. **Ray Cluster**: Use Ray to scale heavy workloads. For example, wrap your retrieval or indexing jobs in Ray tasks to utilize all cluster nodes. Ray's auto-scaler can manage pods on GKE. Ray Serve can expose your RAG chain as a scalable service behind an API.

2. **Kubernetes**: Deploy each component in K8s:
   - LLM model server (with GPUs) – already done in Step 2.
   - Vector DB service (FAISS/Qdrant container or managed service).
   - ADK agent services (each agent as a Deployment + Service).
   - MCP tool servers (as Pods).
   - Any Go microservices (e.g. custom search APIs) – compile them into containers and deploy.
   - MLflow tracking server (optional) – can run on K8s or use managed MLflow on Vertex AI.

3. **Ingress & Security**: Expose only a gateway (e.g. HTTP(S) Ingress) for your front-end or API. Keep everything else internal to the cluster. Use HTTPS and authentication (token or OIDC) for calling the model and agent APIs.

Example: Your ADK agent container might have an entrypoint to start the agent loop (reading user prompts, calling LLM and tools). Behind the scenes it calls MCP servers or other agents (via A2A) as needed.

### Step 9: Track Experiments and Deploy Models with MLflow

1. **Experiment Tracking**: Whenever you adjust your setup (e.g. fine-tune an LLM on code, change prompt templates, or tweak retrieval parameters), log those runs to MLflow. For example:

```python
import mlflow
mlflow.start_run()
mlflow.log_param("model", "mistral-7b")
mlflow.log_param("vector_db", "FAISS")
mlflow.log_metric("exact_match", 0.87)
mlflow.log_metric("average_latency", 1.2)
mlflow.end_run()
```

This lets your team compare which model or setup works best on tasks like "find code examples" or "answer FAQs".

2. **Model Registry**: Once you have a winning model (say, a fine-tuned Llama2), register it in MLflow's Model Registry. Give it a name (e.g. "innersource-assistant") and stage (e.g. "Production"). This serves as the source of truth for deployment.

3. **Integration**: You can even call MLflow's APIs from Go if needed (e.g. a Go service that queries MLflow to check the latest model version). But mostly, MLflow will be a separate service (with UI) that ML engineers use to organize models.

4. **Reproducibility**: Because MLflow logs parameters and code versions, you can always reproduce a run. This is crucial in large organizations to ensure models meet compliance and reliability standards.

### Step 10: Test, Iterate, and Monitor

1. **Testing**: Have real users try the assistant with sample queries (e.g. "Where is the client-service protocol documented?"). Check if answers are accurate and complete. Use ADK's built-in evaluation tools to compare answers to expected responses.

2. **User Feedback**: Collect feedback logs; consider adding a tool for users to flag incorrect answers. Agents can learn by incorporating new docs or retraining embeddings on the latest data.

3. **Monitoring**: Use logs and MLflow to monitor usage and performance (throughput, errors). Ray and K8s provide metrics (CPU, GPU use).

4. **Iteration**: Based on feedback, you might need to refine prompts, add new tools (e.g. Slack integration), or train a custom embedding model on company jargon. Repeat Steps 3–9 as needed to improve.

## Conclusion

By following these steps, you build an LLM-powered system that connects teams and data. The Retrieval-Augmented agent answers questions from company knowledge (improving innersource access), ADK and MCP link the LLM to internal tools, and A2A enables multi-agent workflows. LangChain simplifies integrating LLMs and retrieval, Ray handles scaling, and MLflow keeps your MLOps process organized [mlflow.org](https://mlflow.org/docs/2.8.0/) [cloud.google.com](https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/overview).

This system fosters cross-team collaboration: developers can query it for project docs, managers can find relevant tools, and everyone benefits from a unified "AI librarian" for company knowledge. As Google and Anthropic's leaders emphasize, open protocols like MCP and A2A are the foundation for collaborative AI agents [developers.googleblog.com](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) [github.com](https://github.com/google/A2A). With open-source LLMs and cloud-native deployment, your organization can safely harness these advances for better innersource practices and teamwork.

## Sources

Official docs and announcements from OpenAI (MCP) and Google (A2A, ADK) [openai.github.io](https://openai.github.io/openai-agents-python/mcp/) [developers.googleblog.com](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) [cloud.google.com](https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-multi-system-agents-with-vertex-ai) [python.langchain.com](https://python.langchain.com/docs/concepts/rag/) [langchain.com](https://www.langchain.com/) [cloud.google.com](https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/overview) [mlflow.org](https://mlflow.org/docs/2.8.0/), plus LangChain conceptual guide [python.langchain.com](https://python.langchain.com/docs/concepts/rag/). The embedded figure is adapted from LangChain's explanation of RAG [python.langchain.com](https://python.langchain.com/docs/concepts/rag/).

## References

- [Model context protocol (MCP) - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/)
- [Announcing the Agent2Agent Protocol (A2A) - Google Developers Blog](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [GitHub - google/A2A: An open protocol enabling communication and interoperability between opaque agentic applications](https://github.com/google/A2A)
- [Build and manage multi-system agents with Vertex AI | Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-multi-system-agents-with-vertex-ai)
- [Agent Development Kit](https://google.github.io/adk-docs)
- [Retrieval augmented generation (RAG) | LangChain](https://python.langchain.com/docs/concepts/rag/)
- [LangChain](https://www.langchain.com/)
- [Ray on Vertex AI overview | Google Cloud](https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/overview)
- [MLflow: A Tool for Managing the Machine Learning Lifecycle — MLflow 2.8.0 documentation](https://mlflow.org/docs/2.8.0/)
- [Quickstart - Agent Development Kit](https://google.github.io/adk-docs/get-started/quickstart/)

## InnerSource Knowledge Assistant – Implementation Guide

This guide walks through building an InnerSource knowledge assistant on Kubernetes, using open-source LLMs and an integrated architecture with ADK (Agent Development Kit), LangChain RAG, Ray, and MLflow. We assume your cluster has Kubernetes and MLflow already set up. The guide covers:

- **Data Ingestion & Indexing**: Using Ray to parallelize embedding of internal docs and storing them in a vector DB (FAISS/Qdrant) for RAG.
- **Documentation Agent (ADK)**: Creating a single ADK-based agent that answers queries by retrieving and summarizing internal documentation.
- **Multi-Agent System (A2A)**: Building a root agent that delegates tasks to specialized agents (e.g. a DocsAgent, DevAgent) using ADK's multi-agent patterns and the Agent2Agent protocol.
- **LLM Integration**: Using an open-source LLM (e.g. Mistral or LLaMA2) via HuggingFace/transformers in LangChain and ADK.
- **MCP Integration**: Exposing internal data or APIs via Model Context Protocol (MCP) servers so agents can access restricted resources.
- **Scaling with Ray**: Leveraging Ray for distributed document ingestion/embedding and optional inference scaling.
- **MLflow Tracking**: Instrumenting MLflow to log model usage, prompt calls, and performance metrics for audit and tuning.
- **Security Best Practices**: Using Kubernetes best practices (RBAC, secrets), safe prompt design, and container security.

Each step includes example Python scripts, CLI commands, and Kubernetes YAML snippets, with explanations. This is a hands-on guide for DevSecOps/Cloud Engineers new to LLMOps, balancing beginner-friendliness with production-ready practices.

### 1. Prerequisites & Setup
- **Kubernetes Cluster**: Ensure you have a running K8s cluster and kubectl configured.
- **MLflow Tracking Server**: A deployed MLflow server (URI) to log experiments.
- **Language Models**: We will use an open-source LLM (e.g. Mistral 7B-Instruct or LLaMA2-7B). Ensure you have access (via HuggingFace Hub or local weights).
- **Python Environment**: Install necessary packages in a virtualenv: Ray, LangChain, Transformers, FAISS/Qdrant client, MLflow, and ADK. For example:
```bash
pip install ray[default] langchain transformers sentence-transformers faiss-cpu mlflow google-adk fastapi uvicorn pydantic
```
- **MCP/ADK Tools**: We will use Google's ADK (Agent Development Kit) for agents (pip install google-adk). We also use fastmcp or similar to expose MCP endpoints.
- Finally, plan your architecture: usually, you'll have a vector DB (e.g. a Qdrant or FAISS index) as a service, agents as microservices, and possibly a Ray cluster for heavy computation. You may use Helm charts or Docker/K8s YAML to deploy these components.

### 2. Ingest Internal Documents and Build Vector DB
First, gather your internal documents (docs, FAQs, code snippets, etc.) and embed them into a vector database for retrieval. Ray helps scale this ingestion. We'll use Ray to parallelize embedding and Qdrant (or FAISS) to store vectors for fast semantic search [medium.com](medium.com) [python.langchain.com](python.langchain.com).

**Convert Documents to Text**: Load documents from your file system or repository. For example, using Python:
```python
import glob
from pathlib import Path

doc_paths = glob.glob("/mnt/docs/**/*.txt", recursive=True)  # adjust path/pattern
documents = []
for path in doc_paths:
    with open(path, "r", encoding="utf-8") as f:
        documents.append(f.read())
```

**Initialize Ray**: Launch Ray on your cluster (or locally) for parallel tasks.
```python
import ray
ray.init(address="auto")  # or ray.init() for local, or use ray cluster config
```

**Define an Embedding Worker**: Use a lightweight embedding model (e.g. SentenceTransformer or FastEmbed). Wrap it as a Ray remote actor to enable parallel embedding [medium.com](medium.com).
```python
from sentence_transformers import SentenceTransformer

@ray.remote
class EmbedWorker:
    def __init__(self):
        # Use a sentence-transformer model for embeddings
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

# Example: create 4 workers
workers = [EmbedWorker.remote() for _ in range(4)]
```

**Chunk Documents and Parallel Embed**: Split your documents into chunks, send to workers, and collect embeddings.
```python
chunks = [documents[i::4] for i in range(4)]  # simple round-robin split
embed_tasks = [worker.embed_text.remote(chunk) for worker, chunk in zip(workers, chunks)]
all_embeddings = ray.get(embed_tasks)
# Flatten list of lists:
embeddings = [vec for sublist in all_embeddings for vec in sublist]
```

**Store in Vector DB**: Using Qdrant or FAISS, create a collection and upload embeddings. Example with Qdrant [medium.com](medium.com) [python.langchain.com](python.langchain.com):
```bash
# Deploy Qdrant on Kubernetes (or run via Docker)
# Example YAML (qdrant-deployment.yaml):
apiVersion: apps/v1
kind: Deployment
metadata: { name: qdrant }
spec:
  selector: { matchLabels: { app: qdrant } }
  template:
    metadata: { labels: { app: qdrant } }
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports: [{ containerPort: 6333 }]
---
apiVersion: v1
kind: Service
metadata: { name: qdrant-service }
spec:
  selector: { app: qdrant }
  ports: [{ port: 6333, targetPort: 6333 }]
```

Apply with kubectl apply -f qdrant-deployment.yaml. Then in Python, connect and upload:
```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="qdrant-service:6333")  # or "localhost"
collection = "internal_docs"
client.recreate_collection(
    collection_name=collection,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
)
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"doc_path": doc_paths[i]})
    for i in range(len(embeddings))
]
client.upload_points(collection_name=collection, points=points)
print(f"Stored {len(embeddings)} embeddings in Qdrant.")
```

Citation: LangChain recommends Qdrant as a scalable vector store for semantic search [python.langchain.com](python.langchain.com).

**Verify Retrieval**: You can test a query against the vector DB using cosine similarity:
```python
query = "How to deploy our service on Kubernetes?"
query_emb = SentenceTransformer("all-MiniLM-L6-v2").encode([query]).tolist()
results = client.search(collection, query_vector=query_emb[0], limit=3)
for res in results:
    print(res.payload["doc_path"], "(score:", res.score, ")")
```

At this point you have a vector index of your documents for RAG. The next sections show how to build agents that query this index.

### 3. Documentation Agent with ADK
We now create a single documentation agent using Google's Agent Development Kit (ADK) [developers.googleblog.com](developers.googleblog.com) [google.github.io](google.github.io). This agent will: accept user queries, retrieve relevant context from the vector DB, and use an LLM to generate an answer. ADK makes it easy to wrap LLMs and tools into an agent.

**Install ADK**: (If not done) pip install google-adk.

**Define the LLM**: Use a HuggingFace model (e.g. Mistral) as the language model for the agent. ADK can wrap HuggingFace pipelines. For example:
```python
from google.adk.models import HuggingFaceModel
# Example: Mistral-7B-Instruct via HuggingFace
llm = HuggingFaceModel(model_name="mistralai/Mistral-7B-Instruct-v0.1")
```

**Create a Retrieval Tool**: ADK supports building custom tools. We make a tool that takes a query and returns a combined context from Qdrant.
```python
from google.adk.tools.function import FunctionTool

def retrieve_context(query: str) -> str:
    # Query Qdrant for top-k passages and concatenate
    q_emb = SentenceTransformer("all-MiniLM-L6-v2").encode([query]).tolist()[0]
    results = client.search(collection, query_vector=q_emb, limit=3)
    context = "\n\n".join([open(res.payload["doc_path"]).read() for res in results])
    return context

retrieve_tool = FunctionTool(
    name="InternalDocRetrieval",
    description="Retrieve relevant internal docs for a query",
    func=retrieve_context
)
```

**Assemble the Agent**: Use ADK's LlmAgent with the LLM and the retrieval tool [google.github.io](google.github.io).
```python
from google.adk.agents.llm_agent import LlmAgent

doc_agent = LlmAgent(
    name="DocsAgent",
    llm=llm,
    tools=[retrieve_tool],
    prompt_template=("You are an internal documentation assistant. Use the provided context and answer queries "
                     "clearly.\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"),
)
```

**Run the Agent**: You can now start an interactive session (or deploy via API). For demo:
```python
from google.adk.core.session import Session

session = Session(agents=[doc_agent])
answer = session.send_message("How do I deploy the microservice on Kubernetes?")
print(answer)
```

Under the hood, ADK will call the retrieval tool (via a function call in the prompt), then feed the context + user query to the LLM, returning a result.

ADK is model-agnostic and offers flexible orchestration for multi-agent systems [google.github.io](google.github.io) [google.github.io](google.github.io). Our DocsAgent is a simple LLM agent with one retrieval tool. It uses RAG by explicitly fetching context, which we integrated via the FunctionTool.

### 4. Multi-Agent System (Agent2Agent)
To scale beyond a single agent, we build a multi-agent hierarchy. A root (coordinator) agent delegates tasks to specialized sub-agents (e.g. DocsAgent for docs, DevAgent for code/dev questions). ADK's multi-agent primitives (parent/child agents and tools) enable this [google.github.io](google.github.io) [google.github.io](google.github.io). We also mention the Agent2Agent (A2A) protocol for interoperability (though ADK abstracts much of it).

**Define Specialized Agents**: For example, a DevAgent handles engineering questions (maybe integrated with code search tools):
```python
dev_agent = LlmAgent(
    name="DevAgent",
    llm=llm,
    tools=[retrieve_tool],  # or other dev-specific tools
    prompt_template=("You are a developer assistant. Use context:\n{context}\nQ: {user_query}\nA:"),
)
```

**Create Agent Tools**: We can turn sub-agents into tools for the root. ADK's AgentTool wraps an agent so one agent can call another as a function [google.github.io](google.github.io).
```python
from google.adk.tools import agent_tool

docs_tool = agent_tool.AgentTool(agent=doc_agent)
dev_tool = agent_tool.AgentTool(agent=dev_agent)
```

**Root (Coordinator) Agent**: The root agent receives user queries and decides which sub-agent to invoke. We do this via a prompt that encourages delegation:
```python
root_prompt = """You are a root assistant. Given a user question, decide whether to ask the DocsAgent or DevAgent. 
If the question is about documentation, respond with function call DocsAgent(query). 
If it's about development tasks, use DevAgent(query)."""

root_agent = LlmAgent(
    name="RootAgent",
    llm=llm,
    tools=[docs_tool, dev_tool],
    prompt_template=root_prompt + "\nUser Query: {user_query}\nResponse:",
)
```

**Agent Team Session**: Put the agents into a session (coordinator+sub-agents). ADK will manage passing of messages and state [google.github.io](google.github.io).
```python
session = Session(agents=[root_agent, doc_agent, dev_agent])
response = session.send_message("What is the endpoint for our user API and where is it documented?")
print(response)
```

Here, RootAgent's LLM will generate something like FunctionCall(name='DocsAgent', args={'query': "What is the endpoint for our user API?"}). ADK catches this, runs doc_agent.run_async(query), and returns its answer back to the root's context.

This explicit invocation pattern (AgentTool) is illustrated in ADK docs [google.github.io](google.github.io). The root agent delegates via function calls to its children. You can also use LLM-driven delegation (letting the LLM craft queries to sub-agents) or share state, but explicit AgentTool calls give precise control. In production, you would deploy each agent (or all as one service with internal routing). Kubernetes YAML for agent deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: knowledge-agent }
spec:
  replicas: 2
  selector: { matchLabels: { app: knowledge-agent } }
  template:
    metadata: { labels: { app: knowledge-agent } }
    spec:
      containers:
      - name: agent
        image: myregistry/knowledge-agent:latest  # build from our Python code
        env: [{ name: MLFLOW_TRACKING_URI, value: "http://mlflow:5000" }]
        ports: [{ containerPort: 8000 }] 
---
apiVersion: v1
kind: Service
metadata: { name: knowledge-agent-svc }
spec:
  selector: { app: knowledge-agent }
  ports: [{ port: 80, targetPort: 8000 }]
```

This service could expose a REST endpoint (e.g. using FastAPI) that calls session.send_message.

### 5. Retrieval-Augmented Generation (RAG) with LangChain
We already used manual retrieval in the ADK tools, but LangChain provides higher-level chains for RAG. You can optionally integrate LangChain for simpler code. For example, a LangChain RetrievalQA chain:
```python
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Qdrant

# Wrap HuggingFace model as an LLM
from transformers import pipeline
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device=0)
hf_llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"max_new_tokens": 512})

# Connect LangChain to Qdrant
vectorstore = Qdrant(client=client, collection_name=collection, embeddings=hf_llm)
qa_chain = RetrievalQA(llm=hf_llm, retriever=vectorstore.as_retriever())

query = "How to configure the CI pipeline?"
answer = qa_chain.run(query)
print(answer)
```

This automatically retrieves relevant passages via Qdrant and feeds them to the LLM. For large contexts, chunking and summarization might be needed. (Cited from LangChain docs: Qdrant is recommended for dense retrieval with LangChain [python.langchain.com](python.langchain.com).)

### 6. MCP Servers for Internal APIs
To let agents access internal systems securely, use the Model Context Protocol (MCP) standard [google.github.io](google.github.io). For example, you might expose an internal knowledge base or file server via an MCP-compliant API. ADK's FastMCP can simplify this.

**FastMCP Server**: Write a FastAPI app decorated with MCP tool definitions. Example: exposing a file-read function.
```python
# mcp_server.py
from fastapi import FastAPI
from mcp.fastapi import FastAPI_MCP
from typing import Literal

app = FastAPI()
mcp = FastAPI_MCP(app, title="Internal File MCP")

@mcp.path("/read_file", name="ReadFile")
def read_file(path: str) -> str:
    # Ensure path security (e.g. allow only specific directories)
    allowed_root = Path("/mnt/docs").resolve()
    file_path = (Path(path)).resolve()
    if not str(file_path).startswith(str(allowed_root)):
        raise ValueError("Access denied")
    return file_path.read_text(encoding="utf-8")
```

Deploy this on K8s:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: mcp-server }
spec:
  selector: { matchLabels: { app: mcp } }
  template:
    metadata: { labels: { app: mcp } }
    spec:
      containers:
      - name: mcp
        image: myregistry/mcp-server:latest  # build from mcp_server.py
        ports: [{ containerPort: 8000 }]
---
apiVersion: v1
kind: Service
metadata: { name: mcp-service }
spec:
  selector: { app: mcp }
  ports: [{ port: 8000, targetPort: 8000 }]
```

**Using MCP in Agent**: ADK can call MCP endpoints as tools. Use ADK's MCPTool to wrap the MCP client. Example in agent code:
```python
from google.adk.tools.mcp_tool import MCPTool
file_tool = MCPTool(name="ReadFile", url="http://mcp-service:8000/mcp")
docs_agent.tools.append(file_tool)
```

Now DocsAgent can call ReadFile(path) as a function to read internal docs via MCP. The MCP enforces access control and can use OAuth or IAM.

This setup means LLMs don't access filesystem directly; they go through the safe MCP API. ADK docs note that MCP standardizes LLM-tool communication for data access [google.github.io](google.github.io).

### 7. Scaling with Ray
Beyond ingestion, Ray can also help scale inference or complex pipelines. For example, you might use Ray Serve to deploy the agent model, enabling autoscaling of the LLM inferences behind a HTTP endpoint. A simple Ray Serve example:
```python
from ray import serve

serve.start(detached=True)
@serve.deployment(route_prefix="/agent-query", num_replicas=3)
class AgentService:
    def __init__(self):
        # Load the agent or model here
        self.agent = session

    async def __call__(self, request):
        data = await request.json()
        query = data["query"]
        response = self.agent.send_message(query)
        return {"answer": response}

AgentService.deploy()
```

This would run in Ray and spawn 3 replicas of the service, each loaded with the agent. Kubernetes can host this Ray Serve instance (using ray-operator or manually). Scaling can then be managed by K8s or Ray autoscaler. Ray was also used above for ingestion [medium.com](medium.com), demonstrating dramatic speed-up of embedding tasks by parallelism.

### 8. Tracking with MLflow
Instrument your code to log into MLflow [mlflow.org](mlflow.org) [mlflow.org](mlflow.org). For each user query or batch run:
```python
import mlflow, time

with mlflow.start_run(run_name="DocsAgent_response") as run:
    mlflow.log_param("model", "Mistral-7B")
    mlflow.log_param("agent", "DocsAgent")
    mlflow.log_text(query, "query.txt")
    start = time.time()
    answer = docs_agent.run(user_query=query)
    duration = time.time() - start
    mlflow.log_metric("response_time_sec", duration)
    mlflow.log_text(answer, "answer.txt")
```

You can also log embeddings, retrieval times, or accuracy metrics (if you have ground truth). MLflow supports custom metrics and LLM-as-judge metrics. Track which model versions and prompts yield the best results. For example, log prompt templates or context lengths. Utilize MLflow's UI to compare runs. (MLflow documentation provides examples of logging LLM metrics [mlflow.org](mlflow.org).) Set environment variable MLFLOW_TRACKING_URI to your MLflow server. In Kubernetes, use a ConfigMap or Secret to store this URI or any API keys, and expose to pods.

### 9. Deployment and Operations
**Containerization**: Dockerize your Python code. For example, a Dockerfile:
```dockerfile
FROM python:3.11-slim
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY . /app/
WORKDIR /app
CMD ["uvicorn", "agent_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & push: docker build -t myregistry/knowledge-agent:latest . && docker push myregistry/knowledge-agent:latest. 

**Kubernetes**: Deploy your agent service (as in earlier YAML snippet), along with the vector DB and MCP server. Use Deployment for the agent, Service to expose it, and HorizontalPodAutoscaler if needed (based on CPU/memory, or custom metrics like request rate). 

**CI/CD**: Use GitOps or pipelines to rebuild images on code change. Ensure to scan images for vulnerabilities, use minimal base images, and pin versions.

### 10. Security Best Practices
- **Least Privilege**: Run containers as non-root users. Use PodSecurityContext to drop privileges.
- **Network Policies**: Restrict which services can communicate. For example, only the agent should call Qdrant and MCP service.
- **Kubernetes RBAC**: Use service accounts with limited roles. For instance, an agent should not have admin privileges.
- **Secrets Management**: Store any API keys or credentials (e.g. HuggingFace tokens) in K8s Secrets or Vault, not in code or images.
- **Validate Inputs**: In MCP endpoints, sanitize/validate all parameters (as shown in the read_file example) to prevent path traversal or injection.
- **Monitor & Audit**: Log all agent queries and responses (with MLflow or a logging system). Monitor for anomalies (e.g. private data leaks).
- **Model Safety**: If using LLMs with sensitive data, implement moderation filters or guardrails in prompts to avoid unintended disclosure.
- **Regular Updates**: Keep libraries (ADK, Ray, LangChain, etc.) updated. ADK and Ray are evolving rapidly; watch for security patches.
- **Container Images**: Use official images, scan for vulnerabilities. For example, add --security-opt=no-new-privileges and readOnlyRootFilesystem in Pod spec if possible.

By following these practices, you ensure the knowledge assistant runs reliably in production. ADK's docs include a Security and Safety section [google.github.io](google.github.io) with further recommendations.

### Conclusion
You now have a full-stack LLM-based InnerSource assistant: internal documents are ingested and indexed (with Ray and a vector DB), a documentation agent is implemented via Google's ADK, and a multi-agent architecture (A2A) handles complex queries. Retrieval-Augmented Generation and MCP servers integrate with your internal data, and MLflow tracks model performance and usage. This hands-on architecture is production-ready and modifiable: you can add more specialized agents (e.g. CodeGenAgent), swap in different LLMs, or extend security layers. Key References: ADK simplifies multi-agent development [google.github.io](google.github.io) [google.github.io](google.github.io); LangChain + Qdrant enable RAG for LLMs [python.langchain.com](python.langchain.com); Ray accelerates parallel embedding [medium.com](medium.com). With these building blocks, your DevSecOps team can iteratively improve the system, monitor it through MLflow, and scale as needed.

## Integrating Kafka and Flink with Agent Architecture

```
┌──────────────────────────────┐
│  1.  User / UI (Slack, Web)  │
└──────────────┬───────────────┘
               │  (HTTP / gRPC)
▼ ➊ Root‑Agent  (ADK)
   ├─ uses LangChain‑RAG tool
   ├─ publishes A2A   ➜  Kafka topic  `a2a.tasks`
   ├─ subscribes      ⇠              `a2a.events`
   └─ calls MCP tools ➜  Kafka topic  `mcp.req`
                                   (request/response pattern)
                  ▲                           │
                  │                           ▼
┌─────────────────┴──┐   Flink Job(s)   ┌──────────────┐
│ 2a. Docs‑Agent     │  (stateful CEP)  │ 2b. Dev‑Agent│
│    (ADK + RAG) <───┤  • route tasks   │  (ADK)       │
│                    │  • enrich msgs   │              │
│  • vector‑DB       │  • fan‑out tool  │  • code‑LLM  │
└────────────────────┘                  └──────────────┘
      ▲        ▲                              ▲
      │        │                              │
      │    (Ray Serve)                   (Ray Serve)
      │                                   
      │       ┌────────────────────────────────┐
      │       │   3. Open‑source LLM Pods      │
      └──────▶│   (Mistral / Llama‑2)          │
              │   + HF pipeline + Ray Serve    │
              └────────────────────────────────┘
                             ▲
               embeddings    │ async RPC
                             │
                ┌────────────────────────────┐
                │ 4. Vector Store (Qdrant)   │
                └────────────────────────────┘
                            
                ┌────────────────────────────┐
                │ 5. MCP Servers (File, Git) │
                └────────────────────────────┘
                            
                ┌────────────────────────────┐
                │ 6. MLflow Tracking         │
                └────────────────────────────┘
```

### Message Flow

1. Root‑Agent receives the end‑user prompt → decides whether it can answer itself or must delegate.
2. It publishes a task event (a2a.tasks) to Kafka using the Agent2Agent schema.
3. Real‑time Flink jobs consume a2a.tasks:
   - Validate / enrich metadata
   - Fan‑out to specialized agent topics (tasks.docs, tasks.dev)
   - Attach observability tags, throttling, retries
4. Sub‑agents (Docs‑Agent, Dev‑Agent) subscribe to their topic, process the task, possibly call MCP tools through the request/response queues (mcp.req ➜ mcp.resp) and/or run a LangChain RAG call that queries Qdrant and the LLM served by Ray.
5. Each sub‑agent emits progress events (a2a.events) and a final artifact message holding the answer.
6. The Root‑Agent (or UI) listens to a2a.events, streams updates back to the user.
7. Every prompt/response, vector‑DB recall, latency metric is logged to MLflow for auditing and evaluation.

### Why Kafka + Flink?

| Need in a multi‑agent org‑wide assistant | Kafka/Flink role |
|------------------------------------------|------------------|
| Loose coupling & back‑pressure between dozens of agents running in different namespaces or clusters | Kafka topics decouple sender & receiver; built‑in retention lets late agents replay tasks |
| Observability & audit of every tool call / answer | Flink jobs can branch each stream into ClickHouse/Elastic and report metrics |
| Complex event‑driven coordination (timeouts, retries, fan‑out, aggregation) | Flink's Stateful CEP (Complex‑Event‑Processing) operators model multi‑step agent workflows declaratively |
| Real‑time enrichment (add auth headers, scrub PII, attach vector‑embedding IDs) | Do it in‑stream with Flink before the message reaches an agent |

### Hands‑On Implementation Steps

Below we extend the guide you already have; only the new pieces are shown.

#### 0. Install Kafka & Flink on K8s

```bash
# Kafka with Strimzi
kubectl create namespace streaming
helm repo add strimzi https://strimzi.io/charts/
helm install kafka strimzi/strimzi-kafka-operator -n streaming

# Flink with the Flink K8s Operator
helm repo add flink-operator https://downloads.apache.org/flink/flink-kubernetes-operator-1.6.0/
helm install flink-op flink-operator/flink-kubernetes-operator -n streaming
```

Create minimal Kafka cluster + topics:

```bash
kubectl apply -f - <<'EOF'
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata: { name: agents-cluster, namespace: streaming }
spec:
  kafka:
    version: 3.6.0
    replicas: 3
    listeners:
      - name: external
        port: 9094
        type: nodeport
        tls: false
  zookeeper:
    replicas: 3
EOF
# topics
for t in a2a.tasks a2a.events tasks.docs tasks.dev mcp.req mcp.resp ; do
  kubectl -n streaming exec $(kubectl -n streaming get pods -l strimzi.io/kind=Kafka -o name | head -1) \
    -- bin/kafka-topics.sh --create --topic $t --bootstrap-server localhost:9092 --partitions 6 --replication-factor 3
done
```

#### 1. Agent Code → Kafka Bridges

Use the confluent-kafka Python client inside each agent image:

```python
from confluent_kafka import Producer, Consumer
import json, os, uuid

bootstrap = os.environ["KAFKA_BOOTSTRAP"]

prod = Producer({"bootstrap.servers": bootstrap})
cons = Consumer({
    "bootstrap.servers": bootstrap,
    "group.id": f"agent-{uuid.uuid4()}",
    "auto.offset.reset": "earliest"
})

def publish_task(topic, task_obj):
    prod.produce(topic, json.dumps(task_obj).encode())
    prod.flush()

def subscribe(topic_list):
    cons.subscribe(topic_list)
    while True:
        msg = cons.poll(1.0)
        if msg is None: continue
        task = json.loads(msg.value().decode())
        yield task
```

Root‑Agent example:

```python
task = {
  "id": str(uuid.uuid4()),
  "sender": "root-agent",
  "type": "docs_query",
  "payload": {"query": user_question},
  "ts": time.time()
}
publish_task("a2a.tasks", task)
```

Docs‑Agent loop:

```python
for task in subscribe(["tasks.docs"]):
    query = task["payload"]["query"]
    context = retrieve_context(query)       # LangChain+Qdrant
    answer  = llm(f"{PROMPT}\n{context}\nQ:{query}\nA:")
    publish_task("a2a.events", {
        "in_reply_to": task["id"],
        "sender": "docs-agent",
        "artifact": {"text": answer},
        "ts": time.time()
    })
```

#### 2. Flink Job – Router & Enricher

A simple Flink DataStream job (Scala/Python) that:

- Reads a2a.tasks
- Splits on type field → writes to tasks.docs or tasks.dev
- Adds tracing headers & pushes a copy to an audit sink

Skeleton (PyFlink):

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaSink
from pyflink.common.serialization import SimpleStringSchema
import json, uuid, time

env = StreamExecutionEnvironment.get_execution_environment()

source = KafkaSource.builder()\
    .set_bootstrap_servers(kafka_bootstrap)\
    .set_topics("a2a.tasks")\
    .set_value_only_deserializer(SimpleStringSchema())\
    .build()

ds = env.from_source(source, WatermarkStrategy.no_watermarks(), "kafka-source")

def route(task_json):
    task = json.loads(task_json)
    task["trace_id"] = task.get("trace_id") or str(uuid.uuid4())
    dest = "tasks.docs" if task["type"] == "docs_query" else "tasks.dev"
    return dest, json.dumps(task)

routed = ds.map(route)

sink = KafkaSink.builder()\
    .set_bootstrap_servers(kafka_bootstrap)\
    .set_record_serializer(KafkaRecordSerializationSchema.builder()
          .set_topic_selector(lambda r: r[0])
          .set_value_serialization_schema(SimpleStringSchema())
          .build())\
    .build()

routed.sink_to(sink)
env.execute("agent-router")
```

Submit via the Flink K8s operator:

```bash
kubectl apply -f flink-job-router.yaml   # contains Jar or Python job spec
```

#### 3. MCP Request/Response via Kafka

Wrap your existing FastAPI MCP server with a Kafka gateway (side‑car or code) that:

- Consumes mcp.req
- Executes the tool (e.g. file read)
- Produces the result onto mcp.resp

Agents post a request event and wait for the correlated response (same req_id). This decouples tool latency from the agent's main loop and adds an audit trail.

#### 4. Observability & MLflow

Flink can tee every event into an Elastic/OpenSearch index (agent‑events) and into MLflow via its REST API:

```python
import mlflow
with mlflow.start_run(run_name="router_latency"):
    mlflow.log_metric("router_lag_ms", lag)
```

Agents already log prompts/answers to MLflow (see previous guide). Combine with Flink's aggregated latency metrics for a 360° view.

### Putting It All Together (Timeline)

| Phase | Goal | New work |
|-------|------|----------|
| P‑0 | Core RAG + Doc agent (finished) | – |
| P‑1 | Deploy Kafka & Flink | Strimzi + Flink Op; create topics |
| P‑2 | Instrument agents to publish/consume A2A events | Python confluent‑kafka wrapper |
| P‑3 | Write & deploy Flink router/enricher jobs | PyFlink/Java job JARs |
| P‑4 | Wrap MCP calls through Kafka | Request/response gateway |
| P‑5 | Tune, observe, audit via MLflow & Elastic | Dashboards, alerts |
| P‑6 | Add further agents (Security‑Agent, HR‑Agent…) | Same pattern: topic, router rule, ADK agent |

### Cheat‑Sheet: When Each Piece Activates

| User action | Kafka topic | Flink job | Agent | Tool |
|-------------|-------------|-----------|-------|------|
| Ask "where is DB schema doc?" | a2a.tasks | Router → tasks.docs | Docs‑Agent | LangChain‑RAG (Qdrant) |
| Docs‑Agent needs latest schema PDF | mcp.req | (none) | MCP File server | reads PDF, returns text on mcp.resp |
| Dev asks "generate Go client code" | a2a.tasks | Router → tasks.dev | Dev‑Agent | Code‑LLM (Ray Serve) |
| Dev‑Agent emits progress | a2a.events | Audit → Elastic | Root‑Agent relays to UI | – |

With this pattern you gain loose coupling, replay, audit, and stream processing power without rewriting your agents. Kafka/Flink sit underneath A2A & MCP, giving the protocols a production‑grade backbone.
