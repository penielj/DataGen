# DataGen

An LLM agent that turns plain-English requests into sample user data, served as a REST API.

Built with LangGraph (ReAct agent), LangChain tools, GPT-4o-mini, and FastAPI. Packaged with
Docker and tested in CI with GitHub Actions.

```
POST /generate  {"prompt": "Make 3 users aged 25-35 with company.com emails"}
```
```json
{
  "session_id": "6f1c...",
  "reply": "I generated 3 users with company.com emails, aged 25 to 35.",
  "data": [{"users": [{"id": 1, "firstName": "Alice", "lastName": "Smith",
            "email": "alice.smith@company.com", "username": "alice1482",
            "age": 29, "registeredAt": "2026-03-14T10:22:05"}], "count": 3}]
}
```

## How it works

- **Agent** (`datagen/agent.py`): a LangGraph ReAct agent with three custom tools —
`generate_sample_users`, `write_json`, `read_json` — which it chains on its own (for example,
generate users, then save them).
- **Core logic** (`datagen/tools.py`): plain Python with input validation, so it is unit-tested
without calling OpenAI.
- **Sandboxed file access**: the agent can only read and write `.json` files inside the output
folder. Absolute paths, `../` traversal, and other file types are rejected, so a prompt can't
make it read or overwrite other files on the server.
- **API** (`datagen/api.py`): FastAPI with per-session chat history, so users can refine a
request across multiple turns.
- **Access control**: every endpoint except `/health` requires an `X-API-Key` header matching
`DATAGEN_API_KEY`, so only you can spend OpenAI credits. If the key isn't set, the server refuses
all requests rather than running open. Errors are logged on the server; callers only see a
generic message.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate` | Send a prompt (up to 50 users per request). Pass the returned `session_id` back to continue the conversation. |
| `GET` | `/files/{name}.json` | Download a file the agent saved. |
| `DELETE` | `/sessions/{id}` | Clear a session's history. |
| `GET` | `/health` | Health check (no key needed). |

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" -H "X-API-Key: $DATAGEN_API_KEY" \
  -d '{"prompt": "Make 3 users aged 25-35 with company.com emails"}'
```

Interactive docs are at `/docs` when the server is running.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # then add your OpenAI key and a DATAGEN_API_KEY

uvicorn datagen.api:app --reload   # API at http://localhost:8000/docs
python cli.py                      # or chat in the terminal
```

## Run with Docker

```bash
docker build -t datagen .
docker run -p 8000:8000 --env-file .env datagen
```

## Tests

```bash
pytest -v
```

The tests cover data generation, input validation, the path-sandboxing rules, and the API (with
the LLM replaced by a fake). They run on every push via GitHub Actions.
