from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()



llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)


pdf_path = "data\Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

pages = pdf_loader.load()
print(f"PDF loaded successfully with {len(pages)} pages")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = "./chroma_db"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


vectorstore = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)

print("ChromaDB created successfully!")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


@tool
def retriever_tool(query: str) -> str:
    """
    Search and return information from the Stock Market Performance 2024 document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the document."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

tools = [retriever_tool]

# 🔥 IMPORTANT: bind tools
llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0


system_prompt = """
You are an AI assistant answering questions about Stock Market Performance 2024.

Use the retriever_tool to fetch relevant information from the PDF.
You may call the tool multiple times if needed.
Always cite document sections in your answer.
"""

tools_dict = {tool.name: tool for tool in tools}


def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    response = llm.invoke(messages)
    return {"messages": [response]}


def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query')}")

        if t['name'] not in tools_dict:
            result = "Invalid tool name."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))

        results.append(
            ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            )
        )

    return {"messages": results}


graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== 📊 RAG AGENT (OLLAMA) ===")

    while True:
        user_input = input("\nAsk your question (type 'exit' to quit): ")

        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    running_agent()