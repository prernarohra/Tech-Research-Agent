import os
from semantic_router.encoders import OpenAIEncoder
from datasets import load_dataset
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
from tqdm.auto import tqdm
from typing import TypedDict, Annotated
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
import operator
import re
from langchain_core.tools import tool
from serpapi import GoogleSearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


encoder = OpenAIEncoder(name="text-embedding-3-small")

dataset = load_dataset("vencortex/TechNews", split="train")

pc = Pinecone(api_key="your_api_key")

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"  
)

dims = len(encoder(["some random text"])[0])

index_name = "gpt-4o-research-agent"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=dims,  
        metric='dotproduct',
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)
index.describe_index_stats()

data = dataset.to_pandas().iloc[:10000]

batch_size = 128

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    batch = data[i:i_end].to_dict(orient="records")

    metadata = [{
        "title": r["title"],
        "content": r["text"],
        "arxiv_id": [r["symbol"]]
    } for r in batch]
 
    ids = [r["symbol"] for r in batch]
 
    content = [r["text"] for r in batch]
 
    embeds = encoder(content)
  
    index.upsert(vectors=zip(ids, embeds, metadata))

    class AgentState(TypedDict):
        input: str
        chat_history: list[BaseMessage]
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

symbol = "AAPL"  


articles_by_symbol = [x for x in dataset if x['symbol'] == symbol]

symbol_pattern = re.compile(r"\bAAPL\b")  

articles_with_symbol = [x for x in dataset if re.search(symbol_pattern, x['text']) is not None]

@tool("fetch_technews")
def fetch_technews(symbol: str):
    """Fetches tech news articles related to a specific stock/company symbol from the TechNews dataset."""

    if len(articles_with_symbol) == 0:
        return f"No articles found for symbol: {symbol}"

    first_article = articles_with_symbol[0]
    return {
        "title": first_article['title'],
        "text": first_article['text'],
        "symbol": symbol
    }

serpapi_params = {
    "engine": "google",
    "api_key": os.getenv("SERPAPI_KEY"),
    "num": 5  
    
}

@tool("web_search")
def web_search(query: str):
    """Finds tech knowledge information using Google search. Can also be used
    to augment more 'tech' knowledge to a previous specialist query."""
    sites = ["https://techcrunch.com", "https://gizmodo.com", "https://www.engadget.com", "https://www.wired.com", "https://www.techradar.com", "https://www.cnet.com/", "https://www.digitaltrends.com/", "https://lifehacker.com/"]
   
    search = GoogleSearch({
        **serpapi_params,
        "q": " OR ".join([f"site:{site}" for site in sites]) + query,
        
    })
    results = search.get_dict().get("organic_results", [])
    contexts = "\n---\n".join(
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )
    return contexts

def format_rag_contexts(matches: list):
    contexts = []
    for x in matches:
        revelant = (
            f"Title: {x['metadata']['title']}\n"
        )
        contexts.append(revelant)
    context_str = "\n---\n".join(contexts)
    return context_str

@tool("rag_search_filter")
def rag_search_filter(query: str, symbol: str):
    """Finds information from our tech database using a natural language query
    and a specific symbol. Allows us to learn more details about a specific news."""
    xq = encoder([query])
    xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={"symbol": symbol})
    context_str = format_rag_contexts(xc["matches"])
    return context_str

@tool("rag_search")
def rag_search(query: str):
    """Finds specialist information on AI using a natural language query."""
    xq = encoder([query])
    xc = index.query(vector=xq, top_k=2, include_metadata=True)
    context_str = format_rag_contexts(xc["matches"])
    return context_str

@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
):
    """Returns a natural language response to the user in the form of a research
    report. There are several sections to this report, those are:
    - `introduction`: a short paragraph introducing the user's question and the
    topic we are researching.
    - `research_steps`: a few bullet points explaining the steps that were taken
    to research your report.
    - `main_body`: this is where the bulk of high quality and concise
    information that answers the user's question belongs. It is 3-4 paragraphs
    long in length.
    - `conclusion`: this is a short single paragraph conclusion providing a
    concise but sophisticated view on what was found.
    - `sources`: a bulletpoint list provided detailed sources for all information
    referenced during the research process
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return ""

system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (ie, if the tool appears in the scratchpad twice, do
not use it again).

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected plenty of information
to answer the user's question (stored in the scratchpad) use the final_answer
tool."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key="your_api_key",
    temperature=0
)

tools=[
    rag_search_filter,
    rag_search,
    fetch_technews,
    web_search,
    final_answer
]

def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)

oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

inputs = {
    "input": "what is tesla?",
    "chat_history": [],
    "intermediate_steps": [],
}
out = oracle.invoke(inputs)
out

def run_oracle(state: list):
    print("run_oracle")
    print(f"intermediate_steps: {state['intermediate_steps']}")
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    return {
        "intermediate_steps": [action_out]
    }

def router(state: list):

    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    else:
       
        print("Router invalid format")
        return "final_answer"
    
tool_str_to_func = {
    "rag_search_filter": rag_search_filter,
    "rag_search": rag_search,
    "fetch_technews": fetch_technews,
    "web_search": web_search,
    "final_answer": final_answer
}

def run_tool(state: list):
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    print(f"{tool_name}.invoke(input={tool_args})")
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    return {"intermediate_steps": [action_out]}

graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)
graph.add_node("rag_search_filter", run_tool)
graph.add_node("rag_search", run_tool)
graph.add_node("fetch_technews", run_tool)
graph.add_node("web_search", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")

graph.add_conditional_edges(
    source="oracle",  
    path=router,  
)


for tool_obj in tools:
    if tool_obj.name != "final_answer":
        graph.add_edge(tool_obj.name, "oracle")

graph.add_edge("final_answer", END)

runnable = graph.compile()

out = runnable.invoke({
    "input": "tell me something latest about tesla",
    "chat_history": [],
})

def build_report(output: dict):
    research_steps = output["research_steps"]
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    sources = output["sources"]
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return f"""
INTRODUCTION
------------
{output["introduction"]}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output["main_body"]}

CONCLUSION
----------
{output["conclusion"]}

SOURCES
-------
{sources}
"""

print(build_report(
    output=out["intermediate_steps"][-1].tool_input
))