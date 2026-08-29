import base64
import os
import io
import contextlib
import requests
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import TavilySearchResults, WikipediaQueryRun, tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WikipediaLoader
import wikipedia
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
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
    url: str | None

def build_groq_llm():
    """Build Groq openai/gpt-oss-120b"""
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("Groq API Key not set.")
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def groq_api_key_getter():
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("Groq API Key not set.")
    return groq_key

def transcribe_with_groq_whisper(audio_file_path: str):
    """Send an audio file to Groq Whisper to transcribe and return the transcript."""
    api_key = groq_api_key_getter()
    filename = os.path.basename(audio_file_path)
    ext = Path(filename).suffix.lower()

    with open(audio_file_path, "rb") as file:
        audio_bytes = file.read()
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            files={
                "file": (filename, audio_bytes, "audio/mpeg")
            },
            data={
                "model": "whisper-large-v3",
                "response_format": "text"
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.text.strip()

model = build_groq_llm()

@tool
def add_numbers(a: int, b: int) -> float:
    """Adds two numbers, a and b and returns a float sum."""
    return a + b

@tool
def subtract_numbers(a: int, b: int) -> float:
    """Subtracts two numbers, a by b and returns a float difference."""
    return a - b

@tool
def multiply_numbers(a: int, b: int) -> float:
    """Multiplies two numbers, a by b and returns a float product."""
    return a * b

@tool
def divide_numbers(a: int, b: int) -> float:
    """Divides two numbers, a by b and returns a float quotient."""
    return a / b

"""@tool
def wikipedia_search(query: str):
    Search wikipedia for a query and return a max of three results.
       Takes a string query as the search query
    try:
        search_results = WikipediaLoader(query=query, load_max_docs=3).load()

        if not search_results:
            return f"No Wikipedia results found for {query}. Consider another query or try a web search."
        return "\n\n---\n\n".join(
            f"Title: {doc.metadata.get('title', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in search_results
        )
    except Exception as e:
        print(f"Wikipedia search failed for {query}: {e}")
        return f"Wikipedia search failed for {query}. Try a web search instead."
"""

wikipedia_search_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=5000
    )
)

tavily_key = os.environ.get("TAVILY_API_KEY")
if tavily_key:
    web_search_tool = TavilySearchResults(max_results=5)

@tool
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


@tool
def download_and_read_file(task_id: str) -> str:
    """Download and read the file attached to the GAIA task its contents.
       Always call this first if there is a file attached to a GAIA Task.
       This supports file types csv, txt, py, json, md, ymal, html, xml, xlsx, and mp3.
       USE FOR AUDIO FILES
       
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

        if ext in(".csv", ".txt", ".py", ".json", ".md", ".ymal", ".html", ".xml", ""):
            return response.text

        if ext == ".xlsx" or "xlsx" in content_type:
            import pandas as pd
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as file:
                file.write(response.content)
                temp_path = file.name

            read_file = pd.read_excel(temp_path)
            return read_file.to_string()
        if ext == ".csv" or "csv" in content_type:
            import pandas as pd
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as file:
                file.write(response.content)
                temp_path = file.name
            read_file = pd.read_csv(temp_path)
            return read_file.to_string()
        if ext in(".mp3", ".m4a", ".wav", ".ogg", ".flac" or "audio" in content_type):
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as file:
                file.write(response.content)
                temp_path = file.name

            try:
                return transcribe_with_groq_whisper(temp_path)
            finally:
                os.remove(temp_path)

    except Exception as e:
        return f"error downloading file {e}"

@tool
def get_youtube_video_transcript(url: str) -> str:
    """Gets the transcript/subtitles of a given YouTube video with the provided string.
       Necessary for tasks requiring knowing what is said in a Youtube Video.
       
       One Argument: A string url that is the link for the Youtube video we want the
       transcript from. """

    from youtube_transcript_api import YouTubeTranscriptApi
    from pytube import extract
    ytt = YouTubeTranscriptApi()
    try: 
        video_id = extract.video_id(url)
        # fetch youtube transcript in multiple languages
        video_transcript = ytt.fetch(video_id, languages=['en', 'en_us', 'en_gb'])
        return "\n".join(snippet.text for snippet in video_transcript)

    except Exception as e:
        error_msg = f"Error fetching transcript from url: {url}"
        print(error_msg)
        return ""

@tool
def execute_python_code(code: str) -> str:
    """Execute Python code and return it's given output. Use for math, logic, or any tasks requiring Python code.
    
       Arguments:
            code: a valid string of python code to execute. Valid results must be printed to be read."""
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {})

        return output.getvalue().strip()

    except Exception as e:
        return f"Error executing python code: {code}. Error msg: {e}."
    
tools = [
    add_numbers,
    subtract_numbers,
    divide_numbers,
    multiply_numbers,
    wikipedia_search_tool,
    get_youtube_video_transcript,
    execute_python_code,
    download_and_read_file,
    extract_text_from_image,
    web_search_tool,
]

model_with_tools = model.bind_tools(tools)
with open("systemprompt.txt", "r", encoding="utf-8") as sys_msg:
    system_prompt = sys_msg.read()


def assistant(state: AgentState):
    textual_description_of_tools="""
    Available Tools:

    extract_text_from_image: Extracts text from an image file. Only use this tool if a file_path is currently not None in the State.
       One argument, takes in a string file path, and returns the extracted text from the image.
    
    execute_python_code: Execute Python code and return it's given output. Use for math, logic, or any tasks requiring Python code.
       Arguments:
            code: a valid string of python code to execute. Valid results must be printed to be read.
    
    get_youtube_video_transcript: Gets the transcript/subtitles of a given YouTube video with the provided string.
       Necessary for tasks requiring knowing what is said in a Youtube Video.
       One Argument: A string url that is the link for the Youtube video we want the
       transcript from.       

    download_and_read_file: Download and read the file attached to the GAIA task its contents.
       Always call this first if there is a file attached to a GAIA Task.
       This supports file types csv, txt, py, json, md, ymal, html, xml, xlsx, and mp3.
       USE FOR AUDIO FILES
       
       One argument, the task_id that belongs to the GAIA task whose file should be downloaded and read.
    
    wikipedia_search_tool: Search wikipedia for a query and return a max of three results.
       Takes a string query as the search query
    
    web_search_tool: Search Tavily for a query and return the top 5 search results.
    
    add_numbers: a + b
    subtract_numbers: a - b
    divide_numbers: a / b
    multiply_numbers: a * b
    """
    sys_msg = SystemMessage(content=system_prompt + "\n\n" + textual_description_of_tools)
    return {
        "messages": [model_with_tools.invoke([sys_msg] + state['messages'])],
        "file_path": state["file_path"],
        "task_id": state["task_id"],
        "url": state["url"],
    }

builder = StateGraph(AgentState)

#nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

#edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()


    


    