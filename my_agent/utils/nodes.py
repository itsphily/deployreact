from functools import lru_cache, wraps
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
import asyncio

# Create a cache that works with async functions
def async_lru_cache(maxsize=128):
    cache = {}
    
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = await fn(*args, **kwargs)
            return cache[key]
        return wrapper
    
    return decorator

@async_lru_cache(maxsize=4)
async def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
async def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
async def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = await _get_model(model_name)
    response = await model.ainvoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)