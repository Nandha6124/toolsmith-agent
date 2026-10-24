import os

os.environ["SAMPLE_DB_PATH"] = "./test_agent.db"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import agent as agent_mod  # noqa: E402
from app.agent import run_agent  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm import LLMError, LLMResponse  # noqa: E402
from app.main import app, get_agent_runner  # noqa: E402
from app.seed import seed  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def db():
    settings.sample_db_path = "./test_agent.db"
    seed(settings.sample_db_path)
    yield
    os.remove("./test_agent.db")


def call(name, args, inp=100, out=20):
    part = {"functionCall": {"name": name, "args": args}}
    return LLMResponse(parts=[part], input_tokens=inp, output_tokens=out, function_calls=[(name, args)])


def final(text, inp=100, out=20):
    return LLMResponse(parts=[{"text": text}], input_tokens=inp, output_tokens=out, text=text)


def scripted(*responses):
    it = iter(responses)
    seen = []

    def llm(contents, tools, system):
        seen.append(contents)
        return next(it)

    llm.seen = seen
    return llm


def test_multi_step_task_uses_tools_then_answers():
    llm = scripted(
        call("query_database", {"sql": "SELECT price FROM products WHERE id=1"}),
        call("calculator", {"expression": "1250 * 0.18"}),
        final("GST on the laptop is 225.0"),
    )
    res = run_agent("What is 18% GST on the Laptop Pro 14?", llm=llm)
    assert res.stopped_reason == "finished"
    assert [s.tool for s in res.steps] == ["query_database", "calculator"]
    assert "1250.0" in res.steps[0].observation and res.steps[1].observation == "225.0"
    assert res.answer.startswith("GST") and res.llm_calls == 3 and res.total_tokens == 360
    # tool results were fed back to the model as functionResponse parts
    assert llm.seen[-1][-1]["parts"][0]["functionResponse"]["name"] == "calculator"


def test_tool_error_is_returned_to_model_and_agent_recovers():
    llm = scripted(call("calculator", {"expression": "1/0"}), call("calculator", {"expression": "1/1"}), final("1"))
    res = run_agent("divide", llm=llm)
    assert res.steps[0].observation.startswith("ERROR") and res.stopped_reason == "finished"


def test_max_steps_guardrail(monkeypatch):
    monkeypatch.setattr(settings, "max_steps", 3)
    llm = scripted(*[call("calculator", {"expression": f"{i}+1"}) for i in range(10)])
    res = run_agent("loop forever", llm=llm)
    assert res.stopped_reason == "max_steps" and len(res.steps) == 3 and res.llm_calls == 3


def test_repeated_call_detected():
    llm = scripted(call("calculator", {"expression": "1+1"}), call("calculator", {"expression": "1+1"}))
    res = run_agent("stuck", llm=llm)
    assert res.stopped_reason == "repeated_call" and len(res.steps) == 1


def test_token_budget(monkeypatch):
    monkeypatch.setattr(settings, "max_total_tokens", 500)
    llm = scripted(call("calculator", {"expression": "1+1"}, inp=600), call("calculator", {"expression": "2+2"}))
    res = run_agent("expensive", llm=llm)
    assert res.stopped_reason == "token_budget" and res.llm_calls == 1


def test_cost_budget(monkeypatch):
    monkeypatch.setattr(settings, "max_cost_usd", 0.0001)
    monkeypatch.setattr(settings, "max_total_tokens", 10**9)
    llm = scripted(call("calculator", {"expression": "1+1"}, inp=10_000, out=10_000), final("x"))
    assert run_agent("pricey", llm=llm).stopped_reason == "cost_budget"


def test_llm_error_returns_clean_result():
    def llm(*a):
        raise LLMError("down")

    res = run_agent("hi", llm=llm)
    assert res.stopped_reason == "llm_error" and "down" in res.answer


def test_api_endpoint():
    llm = scripted(call("calculator", {"expression": "6*7"}), final("42"))
    app.dependency_overrides[get_agent_runner] = lambda: (lambda task: agent_mod.run_agent(task, llm=llm))
    client = TestClient(app)
    r = client.post("/agent/run", json={"task": "What is 6 times 7?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "42" and body["steps"][0]["tool"] == "calculator"
    assert body["usage"]["total_tokens"] == 240 and body["usage"]["est_cost_usd"] > 0
    assert client.post("/agent/run", json={"task": "x"}).status_code == 422
    assert {t["name"] for t in client.get("/tools").json()} == {"calculator", "web_search", "query_database"}
    app.dependency_overrides.clear()
