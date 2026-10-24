import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import tools as toolbox
from .config import settings
from .llm import LLMError, LLMResponse, call_model

SYSTEM_PROMPT = """You are a careful assistant that solves tasks by using tools.
- Use calculator for ANY arithmetic. Use query_database for questions about products, customers and orders.
  Use web_search for general facts.
- Think about what you still need, call one or more tools, read the results, and repeat.
- If a tool returns an ERROR, fix your input and try again, or try another approach.
- Do not repeat a search that already gave you the answer; one or two searches are usually enough.
- When you have enough information, answer concisely and mention the numbers you used."""

LLM = Callable[[List[Dict[str, Any]], List[Dict[str, Any]], str], LLMResponse]


@dataclass
class Step:
    step: int
    tool: str
    args: Dict[str, Any]
    observation: str


@dataclass
class AgentResult:
    answer: str
    stopped_reason: str  # finished | max_steps | token_budget | cost_budget | repeated_call | llm_error
    steps: List[Step] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def est_cost_usd(self) -> float:
        return (
            self.input_tokens * settings.price_per_1m_input + self.output_tokens * settings.price_per_1m_output
        ) / 1_000_000


def run_agent(task: str, llm: Optional[LLM] = None) -> AgentResult:
    """Think -> act -> observe loop with hard limits on steps, tokens, cost and repeated calls."""
    llm = llm or call_model
    contents: List[Dict[str, Any]] = [{"role": "user", "parts": [{"text": task}]}]
    result = AgentResult(answer="", stopped_reason="max_steps")
    seen_calls = set()

    for step_no in range(1, settings.max_steps + 1):
        if result.total_tokens >= settings.max_total_tokens:
            result.stopped_reason = "token_budget"
            break
        if result.est_cost_usd >= settings.max_cost_usd:
            result.stopped_reason = "cost_budget"
            break

        try:
            resp = llm(contents, toolbox.DECLARATIONS, SYSTEM_PROMPT)
        except LLMError as exc:
            result.stopped_reason = "llm_error"
            result.answer = f"The language model could not be reached: {exc}"
            return result
        result.llm_calls += 1
        result.input_tokens += resp.input_tokens
        result.output_tokens += resp.output_tokens

        if not resp.function_calls:  # model answered -> done
            result.answer = resp.text
            result.stopped_reason = "finished"
            return result

        contents.append({"role": "model", "parts": resp.parts})
        response_parts = []
        for name, args in resp.function_calls:
            key = (name, json.dumps(args, sort_keys=True))
            if key in seen_calls:  # same call twice = stuck in a loop
                result.stopped_reason = "repeated_call"
                result.answer = result.answer or "Stopped: the agent kept repeating the same tool call."
                return result
            seen_calls.add(key)
            observation = toolbox.run_tool(name, args)
            result.steps.append(Step(step_no, name, args, observation))
            response_parts.append({"functionResponse": {"name": name, "response": {"result": observation}}})
        contents.append({"role": "user", "parts": response_parts})

    if result.stopped_reason == "max_steps":
        # Out of steps: one last call WITHOUT tools so the model summarizes what it found.
        contents.append({"role": "user", "parts": [{"text": "You are out of steps. Using only the information gathered so far, give your best final answer now. Say clearly if something is uncertain."}]})
        try:
            final = llm(contents, [], SYSTEM_PROMPT)
            result.llm_calls += 1
            result.input_tokens += final.input_tokens
            result.output_tokens += final.output_tokens
            if final.text and not final.function_calls:
                result.answer = final.text
        except LLMError:
            pass

    if not result.answer:
        result.answer = f"Stopped before finishing ({result.stopped_reason.replace('_', ' ')})."
        if result.steps:
            last = result.steps[-1]
            result.answer += f" Last result from {last.tool}: {last.observation[:300]}"
    return result
