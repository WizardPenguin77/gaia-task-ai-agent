import base64
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    file_path: str | None
    task_id: str | None

def add_numbers(a: int, b: int) -> float:
    """Adds two numbers, a and b and returns a float sum."""
    return a + b

def subtract_numbers(a: int, b: int) -> float:
    """Subtracts two numbers, a by b and returns a float difference."""
    return a - b

def multiply_numbers(a: int, b: int) -> float:
    """Multiplies two numbers, a by b and returns a float product."""
    return a * b

def divide_numbers(a: int, b: int) -> float:
    """Divides two numbers, a by b and returns a float quotient."""
    return a / b

def wikipedia_search(query: str):
    """Search wikipedia for a query and return a max of three results.
       Takes a string query as the search query"""
    search_results = WikipediaLoader(query=query, load_max_docs=3).load()
    return "\n\n---\n\n".join(
        f"Title: {doc.metadata.get('title', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in search_results
    )

web_search_tool = TavilySearchResults(max_results=5)

def extract_text_from_image(input_file_path: str) -> str:
    """Extracts text from an image file. Only use this tool if a file_path is currently not None in the State.
       One argument, takes in a string file path, and returns the extracted text from the image."""
    extracted_text = ""
    