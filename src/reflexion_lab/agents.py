from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, RuntimeConfig, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @staticmethod
    def _infer_failure_mode(reason: str, reflection_count: int) -> Literal["entity_drift", "incomplete_multi_hop", "wrong_final_answer", "looping", "reflection_overfit"]:
        lowered = reason.lower()
        if "loop" in lowered:
            return "looping"
        if "first hop" in lowered or "second hop" in lowered or "multi-hop" in lowered or "partial" in lowered:
            return "incomplete_multi_hop"
        if "drift" in lowered:
            return "entity_drift"
        if reflection_count >= 2 and ("overfit" in lowered or "still wrong" in lowered):
            return "reflection_overfit"
        if "wrong second-hop" in lowered:
            return "entity_drift"
        return "wrong_final_answer"

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_reason = ""
        for attempt_id in range(1, self.max_attempts + 1):
            answer, actor_tokens, actor_latency = actor_answer(example, attempt_id, self.agent_type, reflection_memory, runtime=self.runtime)
            judge = evaluator(example, answer, runtime=self.runtime)
            token_estimate = actor_tokens + judge.token_usage
            latency_ms = actor_latency + judge.latency_ms
            reflection: ReflectionEntry | None = None

            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge, runtime=self.runtime)
                reflections.append(reflection)
                reflection_memory.append(
                    f"Attempt {attempt_id}: {reflection.lesson} Strategy: {reflection.next_strategy}"
                )
                token_estimate += reflection.token_usage
                latency_ms += reflection.latency_ms

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            traces.append(trace)
            final_answer = answer
            final_score = judge.score
            final_reason = judge.reason
            if judge.score == 1:
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        if final_score == 1:
            failure_mode: Literal["none", "entity_drift", "incomplete_multi_hop", "wrong_final_answer", "looping", "reflection_overfit"] = "none"
        elif self.runtime.mode == "mock":
            failure_mode = FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        else:
            failure_mode = self._infer_failure_mode(final_reason, len(reflections))
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self, runtime: RuntimeConfig | None = None) -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime or RuntimeConfig())

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime: RuntimeConfig | None = None) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime=runtime or RuntimeConfig())
