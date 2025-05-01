# go-with-me-innermind
Building an LLM‐based Collaboration System: A Beginner’s Roadmap
This guide explains key concepts (MCP, Agent2Agent/A2A, ADK, RAG) and tools (LangChain, Ray, MLflow) in simple terms, then shows step-by-step how to build an open‐source LLM system on GCP/Kubernetes. The goal is an enterprise “inner source” assistant that helps teams collaborate, discover projects and tools, and access documentation. We’ll use Python (for most AI code) and Go (for microservices/tools) on Google Kubernetes Engine, and open‐source models like Mistral or LLaMA.
Key Concepts & Tools
Model Context Protocol (MCP)
MCP is an open protocol for feeding external data or tools into an LLM at inference time​
openai.github.io
. In simple terms, MCP is like a USB-C port for AI: it lets your LLM connect to standardized “tool servers” (e.g. a database, file system, or search API) to fetch relevant context. For example, an LLM agent can use an MCP file-system server that exposes your project docs. Using an Agents SDK (like Google’s ADK), you can attach MCP servers so the agent calls them as needed. The MCP spec defines two server types – local (stdio subprocesses) or remote (HTTP/SSE) – which your code can invoke (e.g. using MCPServerStdio to run a filesystem tool)​
openai.github.io
. This way, when the LLM needs extra info (e.g. code snippets or docs), it calls the MCP server rather than hallucinating.
Agent2Agent (A2A) Protocol
Agent2Agent (A2A) is Google’s new open standard for letting multiple AI agents communicate and collaborate​
developers.googleblog.com
​
github.com
. Think of it as a common language for agents: each agent publishes an “Agent Card” (a JSON at /.well-known/agent.json) describing its skills and endpoint, and exposes an HTTP API for tasks. For example, an A2A server is an agent that can receive tasks via a REST API, and an A2A client is any code or agent that calls that API to assign work. A2A tasks go through states (submitted, working, completed, etc.), and can use server‐sent events (SSE) for streaming updates. In practice, A2A lets a coding assistant agent talk to a documentation agent, or a support agent ask a finance agent for data. As Google notes, “A2A is an open protocol that provides a standard way for agents to collaborate with each other, regardless of the underlying framework or vendor”​
developers.googleblog.com
. This is still very new, but it means your teams’ agents (even from different tools) can interoperate on the same network.
Google Agent Development Kit (ADK)
The Agent Development Kit (ADK) is Google’s open‐source Python framework for building and deploying AI agents​
cloud.google.com
. It abstracts away much of the plumbing so you can define agents, tools, workflows and multi-agent systems in Python code. In ADK you create an Agent by giving it a name, a prompt model, instructions, and a list of tool functions. For example, you might write Python functions that search your codebase or fetch wiki docs, and pass them as tools=[search_code, lookup_doc] when instantiating Agent(...). ADK then automatically converts those functions into callable tools for the LLM to use. Importantly, ADK is model- and deployment-agnostic​
google.github.io
​
cloud.google.com
: you choose any LLM (Gemini or open models) and deploy on any platform (local, Cloud Run, Kubernetes). ADK even supports MCP for secure data integration​
cloud.google.com
. In short, ADK makes agent development “feel more like software development”​
google.github.io
: you write Python code for agents and tools, test locally, and then deploy containers or use Vertex AI’s Agent Engine for production.
Retrieval-Augmented Generation (RAG)
RAG is a technique that combines an LLM with an external knowledge source to improve answers​
python.langchain.com
. Here’s how it works: when the user asks a question, the system retrieves (searches) relevant documents or data from a database (e.g. your internal wiki, code repo, support tickets). Then it appends that retrieved content to the LLM’s prompt so the model can ground its answer in real information. This solves the LLM’s “knowledge cutoff” problem by giving it up-to-date facts. For example, if someone asks about the company’s latest API, the RAG system would first pull the API spec from documentation, then feed it into the LLM to generate a precise answer. The diagram below illustrates RAG: 
https://python.langchain.com/docs/concepts/rag/
Figure: RAG process – (1) a Retrieval System searches a knowledge base for relevant documents, then (2) a Language Model (LLM) is prompted with those documents to generate an answer​
python.langchain.com
. RAG is key for our system: by indexing all project wikis, code docs, and files into a searchable database (vector store), the assistant always answers using company knowledge. This makes answers accurate (“based on our docs”) and keeps internal practices consistent.
LangChain, Ray, and MLflow
LangChain: An open-source Python framework for building LLM applications. LangChain provides chains and agents that connect LLM calls with tools, memories, and external data. In simple terms, LangChain lets you “chain together” LLM prompts, calls to APIs, and logic flows. For example, LangChain offers built-in support for RAG: you can create a RetrievalQA chain that takes a question, uses a vectorstore to retrieve docs, and then calls your LLM with the combined prompt. As LangChain’s docs say, it is a “composable framework to build with LLMs”​
langchain.com
. Beginners often use it to rapidly prototype chatbots or assistants that call APIs, store chat history, or perform multi-step reasoning.
Ray: An open-source compute framework for scaling Python workloads, especially ML tasks. Ray lets you distribute and parallelize any Python function across a cluster. For instance, you might use Ray to index millions of documents into your vector store in parallel, or to serve your LLM model at scale. Ray includes libraries like Ray Serve (for model serving), RLlib (RL training), and Tune (hyperparameter tuning). As Google explains, “Ray is an open-source framework for scaling AI and Python applications” and provides the infrastructure for distributed computing​
cloud.google.com
. In practice, you could use a Ray cluster on GKE to run your vector DB and inference nodes, ensuring high throughput.
MLflow: An open-source platform for managing the ML lifecycle. MLflow helps you log experiments, version models, and deploy them. For example, when you fine-tune or benchmark different LLMs on your data, MLflow can log each run’s parameters (model type, prompt variants) and metrics (accuracy, latency). It also provides a Model Registry to store and promote models (e.g. “select best open-model for code search”). As the docs state, “MLflow is an open-source platform… to assist ML teams… ensuring each phase is manageable, traceable, and reproducible”​
mlflow.org
. In our system, use MLflow to track LLM evaluation on company benchmarks (like question-answer accuracy) and to manage which model is in production.
Step-by-Step Implementation
Below is a practical roadmap. We assume no prior LLMOps knowledge. Each Step builds on the previous, from setup to deployment.
Step 1: Set Up Environment (Python, Go, GCP/Kubernetes)
Languages: Use Python for AI code (LangChain agents, vector DB setup, model inference). Use Go for any high-performance services, CLIs or web APIs (e.g. a custom A2A client or a microservice that calls the agent).
GCP & Kubernetes: Create a Google Kubernetes Engine (GKE) cluster. Include GPU nodes (e.g. NVIDIA GPUs) for model hosting. For example:
gcloud container clusters create llm-cluster \
    --zone=us-central1-a --machine-type=n1-standard-8 \
    --accelerator type=nvidia-tesla-t4,count=1 --num-nodes=3
This gives you a scalable K8s cluster. Configure kubectl to interact, and install any necessary services (e.g. Kubernetes Dashboard, ingress controller).
Container Registry: Prepare a Container Registry or Artifact Registry on GCP to store your Docker images (for the LLM server, agent server, etc.).
Permissions: Ensure K8s nodes have access to any required GCP services (e.g. permissions to pull data from Cloud Storage if you store docs there).
Step 2: Deploy an Open-Source LLM
Choose a model: Pick a high-quality open model like Mistral-7B or Meta’s Llama-2. For example, Mistral-7B Chat can be fine for English text. Download the model weights from Hugging Face or the provider.
Containerize the model: Create a Docker image that loads the model and exposes an HTTP/gRPC endpoint. You can use libraries like vLLM or text-generation-webui inside the container to serve the model. For instance, a simple Flask app could load Llama2 and respond to /generate requests.
Deploy on K8s: Define a Kubernetes Deployment (and a Service) for the model server. Example (YAML simplified):
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
Testing: Verify you can hit the model endpoint. For example, in Python:
import requests
resp = requests.post("http://llm-serving.default.svc.cluster.local/generate",
                     json={"prompt": "Hello, world!"})
print(resp.json())
Replace with your actual service address. Ensure latency is acceptable; autoscale with an HPA if needed.
Step 3: Prepare the Knowledge Base (Innersource Data)
Gather docs & data: Collect all relevant internal documentation: code repositories, READMEs, wikis, design docs, support tickets, etc. For innersource, include “light” community docs like how-to guides.
Text processing: Clean and split text into chunks (e.g. by paragraph or section). For code, you might extract docstrings or comments.
Embeddings & Vector DB: Compute vector embeddings for each chunk using a sentence-embedding model (open-source, e.g. [sentence-transformers]). Store embeddings in a vector database like Qdrant, Weaviate, or Chroma. For example, in Python with LangChain:
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

docs = [Document(page_content=text, metadata={...}) for text in document_chunks]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
# Save index to disk or persistent storage
vector_store.save_local("inner_source_index")
Deploy the DB: Run the vector DB on K8s (many support container deployment). Ensure it’s accessible by your LLM pipeline. You may also use Google’s managed Vertex AI Matching Engine for large scale.
Step 4: Implement Retrieval-Augmented Generation (RAG)
With the data prepared, build the RAG pipeline. We will use LangChain for simplicity:
Setup LangChain chain: Use a retrieval chain that fetches documents then calls the LLM. For example:
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceLLM

# Load vector store (from Step 3)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = HuggingFaceLLM(model_name="mistral-7b", temperature=0.0)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
Querying: When a user asks a question, run the chain:
answer = qa_chain({"query": "How to onboard new contributors?"})
print(answer["result"])
LangChain will (1) retrieve top-5 relevant docs about onboarding, then (2) prompt the LLM with those docs plus the question. This produces an answer grounded in your innersource docs.
In practice: Deploy this as a service (e.g. a FastAPI app) on K8s so your agents or UI can call it. If you’re concerned about latency, use Ray Serve to host the chain with autoscaling. Ray can parallelize heavy workloads like embedding large queries or serving many requests.
Step 5: Develop Agents with ADK
Use the Agent Development Kit to create one or more intelligent agents. For example, you might build an agent that handles developer queries, and another for project discovery.
Install ADK: In your Python environment: pip install google-adk.
Define tool functions: Write Python functions for the agent’s tools. Examples: search_codebase(query), lookup_doc(query), list_projects(), etc. Each returns structured results (e.g. status + data).
Create an Agent: In code, tie it together. For instance:
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
Run the agent: The ADK SDK provides ways to run the agent locally or in a dev UI. You can feed it user messages, and it will autonomously call the appropriate tools (using your functions) as needed.
State and Memory: ADK supports sessions and memory; you can choose to remember past interactions or not, depending on your use case. For example, you could enable an in-memory or persistent session so the agent “remembers” ongoing tasks.
(See the earlier ADK Quickstart example​
google.github.io
 for a template: define tool functions, then pass them into Agent(...).)
Step 6: Integrate Tools via MCP
To securely connect agents with your data sources, use MCP with your ADK agents:
MCP Servers: Deploy MCP tool servers for anything that requires external context. For example, you could run @modelcontextprotocol/server-filesystem as a subprocess or container to give file access. Or write a simple MCP HTTP server that wraps a proprietary search API.
Connect to Agent: In ADK, add MCP servers to your agent’s runtime. Pseudocode:
from openai_agents.mcp import MCPServerStdio

# Example: local filesystem MCP server (Node.js)
mcp_server = await MCPServerStdio(params={
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data/docs"]
})
my_agent.add_mcp_server(mcp_server)
When the agent runs, it will auto-call list_tools() on the MCP server so it knows what tools are available (e.g. “read file”, “search folder”). If the LLM decides to use one, the SDK will call_tool() on the MCP server.
Usage: Now your agent can say, for example, “Use your file tool to open the architecture doc.” and it will fetch that data from the MCP server. This tightly couples LLM reasoning with actual company data.
Step 7: Enable Multi-Agent Collaboration (Agent2Agent)
If you want separate agents to work together (for example, one agent for frontend devs and another for backend devs), use the A2A protocol:
Run A2A servers: Each agent deployment (e.g. as a container) should host an HTTP endpoint implementing A2A (many frameworks will add this). Expose .well-known/agent.json describing its capabilities.
Discovery & Tasks: One agent (the client) can fetch another’s Agent Card to learn how to contact it. Then it initiates a task via A2A’s /tasks/send API with a message. For example, a “project lookup” agent could ask a “data team” agent: “find all docs about data lakes”. The A2A server on the data agent would receive this task and process it, streaming back results.
Example flow: (Using the GitHub A2A library)
Agent A discovers Agent B’s endpoint and reads its Agent Card (capabilities).
Agent A (A2A client) sends a JSON tasks/send request to Agent B’s /tasks/send URL.
Agent B processes the task, possibly involving its own LLM or tools.
If it’s long-running, Agent B streams progress via SSE to Agent A.
Why A2A: This lets teams run agents independently but still have them collaborate. For example, a security agent (on Go) and a dev agent (on Python) can interoperate. As Google notes, A2A gives a “common language” so agents can share info across ecosystems​
github.com
. Note: A2A is very new, so starting with one agent or simple HTTP calls may suffice initially.
Step 8: Scale and Orchestrate with Ray and Kubernetes
Ray Cluster: Use Ray to scale heavy workloads. For example, wrap your retrieval or indexing jobs in Ray tasks to utilize all cluster nodes. Ray’s auto-scaler can manage pods on GKE. Ray Serve can expose your RAG chain as a scalable service behind an API.
Kubernetes: Deploy each component in K8s:
LLM model server (with GPUs) – already done in Step 2.
Vector DB service (FAISS/Qdrant container or managed service).
ADK agent services (each agent as a Deployment + Service).
MCP tool servers (as Pods).
Any Go microservices (e.g. custom search APIs) – compile them into containers and deploy.
MLflow tracking server (optional) – can run on K8s or use managed MLflow on Vertex AI.
Ingress & Security: Expose only a gateway (e.g. HTTP(S) Ingress) for your front-end or API. Keep everything else internal to the cluster. Use HTTPS and authentication (token or OIDC) for calling the model and agent APIs.
Example: Your ADK agent container might have an entrypoint to start the agent loop (reading user prompts, calling LLM and tools). Behind the scenes it calls MCP servers or other agents (via A2A) as needed.
Step 9: Track Experiments and Deploy Models with MLflow
Experiment Tracking: Whenever you adjust your setup (e.g. fine-tune an LLM on code, change prompt templates, or tweak retrieval parameters), log those runs to MLflow. For example:
import mlflow
mlflow.start_run()
mlflow.log_param("model", "mistral-7b")
mlflow.log_param("vector_db", "FAISS")
mlflow.log_metric("exact_match", 0.87)
mlflow.log_metric("average_latency", 1.2)
mlflow.end_run()
This lets your team compare which model or setup works best on tasks like “find code examples” or “answer FAQs”.
Model Registry: Once you have a winning model (say, a fine-tuned Llama2), register it in MLflow’s Model Registry. Give it a name (e.g. “innersource-assistant”) and stage (e.g. “Production”). This serves as the source of truth for deployment.
Integration: You can even call MLflow’s APIs from Go if needed (e.g. a Go service that queries MLflow to check the latest model version). But mostly, MLflow will be a separate service (with UI) that ML engineers use to organize models.
Reproducibility: Because MLflow logs parameters and code versions, you can always reproduce a run. This is crucial in large organizations to ensure models meet compliance and reliability standards.
Step 10: Test, Iterate, and Monitor
Testing: Have real users try the assistant with sample queries (e.g. “Where is the client-service protocol documented?”). Check if answers are accurate and complete. Use ADK’s built-in evaluation tools to compare answers to expected responses.
User Feedback: Collect feedback logs; consider adding a tool for users to flag incorrect answers. Agents can learn by incorporating new docs or retraining embeddings on the latest data.
Monitoring: Use logs and MLflow to monitor usage and performance (throughput, errors). Ray and K8s provide metrics (CPU, GPU use).
Iteration: Based on feedback, you might need to refine prompts, add new tools (e.g. Slack integration), or train a custom embedding model on company jargon. Repeat Steps 3–9 as needed to improve.
Conclusion
By following these steps, you build an LLM-powered system that connects teams and data. The Retrieval-Augmented agent answers questions from company knowledge (improving innersource access), ADK and MCP link the LLM to internal tools, and A2A enables multi-agent workflows. LangChain simplifies integrating LLMs and retrieval, Ray handles scaling, and MLflow keeps your MLOps process organized​
mlflow.org
​
cloud.google.com
. This system fosters cross-team collaboration: developers can query it for project docs, managers can find relevant tools, and everyone benefits from a unified “AI librarian” for company knowledge. As Google and Anthropic’s leaders emphasize, open protocols like MCP and A2A are the foundation for collaborative AI agents​
developers.googleblog.com
​
github.com
. With open-source LLMs and cloud-native deployment, your organization can safely harness these advances for better innersource practices and teamwork. Sources: Official docs and announcements from OpenAI (MCP) and Google (A2A, ADK)​
openai.github.io
​
developers.googleblog.com
​
cloud.google.com
​
python.langchain.com
​
langchain.com
​
cloud.google.com
​
mlflow.org
, plus LangChain conceptual guide​
python.langchain.com
. The embedded figure is adapted from LangChain’s explanation of RAG​
python.langchain.com
.
Citations
Model context protocol (MCP) - OpenAI Agents SDK

https://openai.github.io/openai-agents-python/mcp/
Model context protocol (MCP) - OpenAI Agents SDK

https://openai.github.io/openai-agents-python/mcp/
Favicon
Announcing the Agent2Agent Protocol (A2A) - Google Developers Blog

https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
Favicon
GitHub - google/A2A: An open protocol enabling communication and interoperability between opaque agentic applications.

https://github.com/google/A2A
Favicon
Build and manage multi-system agents with Vertex AI | Google Cloud Blog

https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-multi-system-agents-with-vertex-ai
Favicon
Agent Development Kit

https://google.github.io/adk-docs
Favicon
Build and manage multi-system agents with Vertex AI | Google Cloud Blog

https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-multi-system-agents-with-vertex-ai
Favicon
Retrieval augmented generation (RAG) | ️ LangChain

https://python.langchain.com/docs/concepts/rag/
Favicon
LangChain

https://www.langchain.com/
Favicon
Ray on Vertex AI overview  |  Google Cloud

https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/overview
Favicon
MLflow: A Tool for Managing the Machine Learning Lifecycle — MLflow 2.8.0 documentation

https://mlflow.org/docs/2.8.0/
Favicon
Quickstart - Agent Development Kit

https://google.github.io/adk-docs/get-started/quickstart/
Favicon
GitHub - google/A2A: An open protocol enabling communication and interoperability between opaque agentic applications.

https://github.com/google/A2A
Favicon
Announcing the Agent2Agent Protocol (A2A) - Google Developers Blog

https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
All Sources
openai.github
Favicondevelope...oogleblog
Favicongithub
Faviconcloud.google
Favicongoogle.github
Faviconpython.langchain
Faviconlangchain
Faviconmlflow
