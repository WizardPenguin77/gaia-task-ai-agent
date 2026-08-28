import base64
import requests
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from app import model
from pathlib import Path
import tempfile

# constants
API_URL = "https://agents-course-unit4-scoring.hf.space"
QUESTIONS_URL = f"{API_URL}/questions"
FILES_URL = f"{API_URL}/files"
SUBMIT_URL = f"{API_URL}/submit"

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
    try:
        # read image and encode as base64
        with open(input_file_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.encode(image_bytes).decode("utf-8")

        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this image. Return only the extracted text. No explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            )
        ]

        response = model.invoke(message)
        extracted_text += response.content + "\n\n"
        return extracted_text.strip()

    except Exception as e:
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""

def download_and_read_file(task_id: str) -> str:
    """Download and read the file attached to the GAIA task its contents.
       Always call this first if there is a file attached to a GAIA Task.
       This supports file types csv, 
       
       One argument, the task_id that belongs to the GAIA task whose file should be downloaded and read."""

    url = f"{FILES_URL}/{task_id}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        #get the file extension from the header
        content_disposition = response.headers.get("content-disposition", "")
        content_type = response.headers.get("content-type", "")
        filename = None
        if "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[1].strip('"')

        if not filename:
            filename = f"{task_id}.bin"

        ext = Path(filename).suffix.lower()

        if ext in{
            ".csv", ".txt", ".py", ".json", ".md", ".ymal", ".html", ".xml", ""
        }:
            return response.text

        if ext == ".xlsx":
            import pandas as pd
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as file:
                file.write(response.content)
                temp_path = file.name

            read_file = pd.read_excel(temp_path)
            return read_file.to_string()
        if ext == ".csv":
            import pandas as pd
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as file:
                file.write(response.content)
                temp_path = file.name
            read_file = pd.read_csv(temp_path)
            return read_file.to_string()

    except Exception as e:
        return f"error downloading file {e}"


def get_youtube_video_transcript(url: str) -> str:
    """Gets the transcript/subtitles of a given YouTube video with the provided string.
       Necessary for tasks requiring knowing what is said in a Youtube Video.
       
       One Argument: A string url that is the link for the Youtube video we want the
       transcript from. """
    
    


    


    