from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

load_dotenv()



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]



@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b 
@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b
@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]


hf_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0,
    max_new_tokens=512
)
chat_model = ChatHuggingFace(llm=hf_llm)
model = chat_model.bind_tools(tools)



def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant. Use tools when needed to solve math operations."
    )
    response = model.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"




graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {
    "messages": [
        ("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")
    ]
}

print_stream(app.stream(inputs, stream_mode="values"))
