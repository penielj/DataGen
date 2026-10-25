"""Agent helper tests that don't call the LLM."""
import json

from langchain_core.messages import AIMessage, ToolMessage

from datagen import agent
from datagen.agent import extract_generated_data, generate_sample_users, run_agent


def test_tool_returns_json_string():
    out = generate_sample_users.invoke({
        "first_names": ["Ana"], "last_names": ["Lee"], "domains": ["a.com"],
        "min_age": 20, "max_age": 30,
    })
    assert json.loads(out)["count"] == 1


def test_extract_generated_data_keeps_only_successful_user_results():
    good = {"users": [{"id": 1}], "count": 1}
    messages = [
        AIMessage(content="thinking"),
        ToolMessage(content=json.dumps(good), name="generate_sample_users", tool_call_id="1"),
        ToolMessage(content=json.dumps({"error": "bad"}), name="generate_sample_users",
                    tool_call_id="2"),
        ToolMessage(content="Successfully wrote", name="write_json", tool_call_id="3"),
        ToolMessage(content="not json", name="generate_sample_users", tool_call_id="4"),
    ]
    assert extract_generated_data(messages) == [good]


def test_run_agent_hides_error_details_but_logs_them(monkeypatch, caplog):
    def broken_agent():
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(agent, "get_agent", broken_agent)
    reply, data = run_agent("hi", [])
    assert "secret internal detail" not in reply.content
    assert data == []
    assert "secret internal detail" in caplog.text
