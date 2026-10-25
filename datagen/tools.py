"""Core DataGen logic: data generation and sandboxed JSON file access.

Everything here is plain Python with no LLM dependencies, so it can be
unit-tested without an OpenAI key. agent.py wraps these as LangChain tools.
"""
import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# The LLM has to pass every name in, and echo the result back to save it,
# so large batches get truncated or expensive long before this limit.
MAX_USERS = 50


class PathNotAllowedError(ValueError):
    """Raised when a requested file path escapes the output directory."""


def get_output_dir() -> Path:
    """Directory that all reads and writes are confined to."""
    out = Path(os.getenv("DATAGEN_OUTPUT_DIR", "output")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_path(filename: str) -> Path:
    """Resolve a filename inside the output directory, rejecting anything else.

    Blocks absolute paths, '..' traversal, subdirectories, and non-.json files
    so an LLM (or an API caller) can't read or overwrite arbitrary files.
    """
    if not filename or not isinstance(filename, str):
        raise PathNotAllowedError("Filename must be a non-empty string.")
    name = Path(filename)
    if name.is_absolute() or len(name.parts) != 1 or name.name in (".", ".."):
        raise PathNotAllowedError(
            f"'{filename}' is not allowed. Use a plain file name like 'users.json'."
        )
    if name.suffix.lower() != ".json":
        raise PathNotAllowedError("Only .json files are allowed.")

    out_dir = get_output_dir()
    resolved = (out_dir / name).resolve()
    if resolved.parent != out_dir:
        raise PathNotAllowedError(f"'{filename}' resolves outside the output directory.")
    return resolved


def save_json(filename: str, data: dict) -> str:
    """Write data as pretty-printed JSON to the output directory."""
    try:
        path = safe_path(filename)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        path.write_text(text, encoding="utf-8")
        return f"Successfully wrote JSON data to '{path.name}' ({len(text)} characters)."
    except PathNotAllowedError as e:
        return f"Error: {e}"
    except (TypeError, ValueError) as e:
        return f"Error: data is not JSON-serializable - {e}"
    except OSError as e:
        return f"Error writing JSON: {e}"


def load_json(filename: str) -> str:
    """Read a JSON file from the output directory and return it as a string."""
    try:
        path = safe_path(filename)
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2)
    except PathNotAllowedError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: File '{filename}' not found."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {e}"
    except OSError as e:
        return f"Error reading JSON: {e}"


def generate_users(
    first_names: List[str],
    last_names: List[str],
    domains: List[str],
    min_age: int,
    max_age: int,
    seed: Optional[int] = None,
) -> dict:
    """Generate sample user records. One user per first name.

    Last names and domains cycle if shorter than first_names. Returns
    {'users': [...], 'count': n} or {'error': message}.
    """
    if not first_names:
        return {"error": "first_names list cannot be empty"}
    if not last_names:
        return {"error": "last_names list cannot be empty"}
    if not domains:
        return {"error": "domains list cannot be empty"}
    if len(first_names) > MAX_USERS:
        return {"error": f"cannot generate more than {MAX_USERS} users at once"}
    if min_age < 0 or max_age < 0:
        return {"error": "ages must be non-negative"}
    if min_age > max_age:
        return {"error": f"min_age ({min_age}) cannot be greater than max_age ({max_age})"}

    rng = random.Random(seed)
    now = datetime.now()
    users = []
    seen_emails = set()

    for i, first in enumerate(first_names):
        last = last_names[i % len(last_names)]
        domain = domains[i % len(domains)]

        # Repeated names would otherwise produce duplicate emails.
        base = f"{first.lower()}.{last.lower()}".replace(" ", "")
        email = f"{base}@{domain}"
        n = 2
        while email in seen_emails:
            email = f"{base}{n}@{domain}"
            n += 1
        seen_emails.add(email)

        users.append({
            "id": i + 1,
            "firstName": first,
            "lastName": last,
            "email": email,
            "username": f"{first.lower().replace(' ', '')}{i + 1}{rng.randint(100, 999)}",
            "age": rng.randint(min_age, max_age),
            "registeredAt": (now - timedelta(days=rng.randint(1, 365))).isoformat(),
        })

    return {"users": users, "count": len(users)}
