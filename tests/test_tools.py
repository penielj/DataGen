import json
import re

import pytest

from datagen.tools import (
    MAX_USERS,
    PathNotAllowedError,
    generate_users,
    load_json,
    safe_path,
    save_json,
)


# ---------- generate_users ----------

def make(**overrides):
    args = dict(first_names=["Ana", "Ben", "Cal"], last_names=["Lee"],
                domains=["a.com", "b.com"], min_age=20, max_age=30, seed=1)
    args.update(overrides)
    return generate_users(**args)


def test_one_user_per_first_name():
    result = make()
    assert result["count"] == 3
    assert [u["firstName"] for u in result["users"]] == ["Ana", "Ben", "Cal"]


def test_last_names_and_domains_cycle():
    users = make()["users"]
    assert {u["lastName"] for u in users} == {"Lee"}
    assert [u["email"].split("@")[1] for u in users] == ["a.com", "b.com", "a.com"]


def test_ages_within_range():
    users = make(first_names=["X"] * 50, min_age=18, max_age=21)["users"]
    assert all(18 <= u["age"] <= 21 for u in users)


def test_ids_are_sequential():
    assert [u["id"] for u in make()["users"]] == [1, 2, 3]


def test_email_format():
    pattern = re.compile(r"^[a-z]+\.[a-z]+\d*@[a-z.]+$")
    assert all(pattern.match(u["email"]) for u in make()["users"])


def test_duplicate_names_get_unique_emails_and_usernames():
    users = make(first_names=["Sam", "Sam", "Sam"], domains=["x.com"])["users"]
    assert len({u["email"] for u in users}) == 3
    assert len({u["username"] for u in users}) == 3


def test_seed_makes_output_reproducible():
    a, b = make(seed=42)["users"], make(seed=42)["users"]
    assert [u["age"] for u in a] == [u["age"] for u in b]


@pytest.mark.parametrize("overrides, message", [
    ({"first_names": []}, "first_names"),
    ({"last_names": []}, "last_names"),
    ({"domains": []}, "domains"),
    ({"min_age": 40, "max_age": 30}, "cannot be greater"),
    ({"min_age": -1}, "non-negative"),
    ({"first_names": ["A"] * (MAX_USERS + 1)}, "more than"),
])
def test_validation_errors(overrides, message):
    result = make(**overrides)
    assert "error" in result
    assert message in result["error"]


# ---------- safe_path ----------

@pytest.mark.parametrize("bad", [
    "../secrets.json",
    "/etc/passwd.json",
    "sub/dir.json",
    "..",
    "",
    "notes.txt",
    ".env",
])
def test_safe_path_rejects_unsafe_names(bad):
    with pytest.raises(PathNotAllowedError):
        safe_path(bad)


def test_safe_path_stays_in_output_dir(isolated_output_dir):
    assert safe_path("users.json").parent == isolated_output_dir.resolve()


# ---------- save_json / load_json ----------

def test_save_then_load_round_trip():
    data = {"users": [{"id": 1, "name": "José"}]}
    assert save_json("users.json", data).startswith("Successfully")
    assert json.loads(load_json("users.json")) == data


def test_save_rejects_traversal_without_writing(tmp_path):
    msg = save_json("../escape.json", {"a": 1})
    assert msg.startswith("Error")
    assert not (tmp_path / "escape.json").exists()


def test_load_missing_file():
    assert "not found" in load_json("nope.json")


def test_load_invalid_json(isolated_output_dir):
    isolated_output_dir.mkdir(parents=True, exist_ok=True)
    (isolated_output_dir / "bad.json").write_text("{not json")
    assert "Invalid JSON" in load_json("bad.json")
