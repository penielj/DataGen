"""LangGraph ReAct agent that uses the DataGen tools."""
import json
import logging
from functools import lru_cache
from typing import List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from datagen import tools as core

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"

SYSTEM_MESSAGE = (
    "You are DataGen, a helpful assistant that generates sample data for applications. "
    "To generate users, you need: first_names (list), last_names (list), domains (list), "
    "min_age, max_age. Fill in any of these values yourself without asking for them. "
    f"You can generate at most {core.MAX_USERS} users per request. "
    "When asked to save users, first generate them with the tool, then immediately call "
    "write_json with the result. File names must be plain names ending in .json, "
    "like 'users.json'. If the user refers to 'those users' from a previous request, "
    "ask them to specify the details again."
)


@tool
def write_json(filename: str, data: dict) -> str:
    """Write a dictionary as pretty-printed JSON to a file in the output folder."""
    return core.save_json(filename, data)


@tool
def read_json(filename: str) -> str:
    """Read and return the contents of a JSON file from the output folder."""
    return core.load_json(filename)


@tool
def generate_sample_users(
    first_names: List[str],
    last_names: List[str],
    domains: List[str],
    min_age: int,
    max_age: int,
) -> str:
    """Generate sample user data. Count is determined by the length of first_names.

    Args:
        first_names: List of first names (one per user)
        last_names: List of last names (will cycle if fewer than first_names)
        domains: List of email domains (will cycle through)
        min_age: Minimum age for users
        max_age: Maximum age for users

    Returns:
        JSON string with a 'users' array, or an 'error' message
    """
    # Returned as a JSON string so the API can parse it back out reliably.
    return json.dumps(core.generate_users(first_names, last_names, domains, min_age, max_age))


TOOLS = [write_json, read_json, generate_sample_users]


@lru_cache(maxsize=1)
def get_agent():
    """Build the agent on first use, so importing this module needs no API key."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    llm = ChatOpenAI(model=MODEL, temperature=0)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)


def extract_generated_data(messages: List[BaseMessage]) -> List[dict]:
    """Pull every successful generate_sample_users result out of a run."""
    results = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "generate_sample_users":
            try:
                parsed = json.loads(msg.content)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict) and "users" in parsed:
                results.append(parsed)
    return results


def run_agent(user_input: str, history: List[BaseMessage]) -> Tuple[AIMessage, List[dict]]:
    """Run one turn. Returns the agent's final reply and any data it generated."""
    try:
        result = get_agent().invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50},
        )
        new_messages = result["messages"][len(history):]
        return result["messages"][-1], extract_generated_data(new_messages)
    except Exception:  # log details server-side; never send raw errors to the caller
        logger.exception("Agent run failed")
        return (
            AIMessage(content="Sorry, something went wrong generating that. "
                              "Please try rephrasing your request."),
            [],
        )
