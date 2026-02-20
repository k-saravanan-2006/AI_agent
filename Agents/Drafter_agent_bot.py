from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

load_dotenv()



document_content = ""



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]



@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document updated successfully!\n\nCurrent content:\n{document_content}"


@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process."""
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(document_content)

        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document saved successfully to '{filename}'."

    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [update, save]



model = ChatOllama(
    model="llama3.2",  # must match ollama list
    temperature=0
).bind_tools(tools)



def our_agent(state: AgentState) -> AgentState:
    global document_content

    system_prompt = SystemMessage(content=f"""
You are Drafter, a helpful writing assistant.

Rules:
- If user wants to create or modify document, use the 'update' tool.
- Always send the FULL updated document when using update.
- If user wants to save and finish, use the 'save' tool.
- After each update, show the current document content.

Current document content:
{document_content}
""")

    if not state["messages"]:
        user_message = HumanMessage(
            content="I'm ready to help you create or edit a document. What would you like to write?"
        )
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")

    if response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {
        "messages": list(state["messages"]) + [user_message, response]
    }



def should_continue(state: AgentState) -> str:
    messages = state["messages"]

    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower():
            return "end"

    return "continue"



def print_messages(messages):
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT:\n{message.content}")



graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()



def run_document_agent():
    print("\n===== 📄 DRAFTER (OLLAMA) =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n===== ✅ DRAFTER FINISHED =====")



if __name__ == "__main__":
    run_document_agent()
