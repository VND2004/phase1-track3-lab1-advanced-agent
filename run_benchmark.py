from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.mock_runtime import RuntimeConfig
from src.reflexion_lab.schemas import AttemptTrace, QAExample, RunRecord
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

if load_dotenv is not None:
    load_dotenv()

app = typer.Typer(add_completion=False)


def _error_record(agent_type: str, example: QAExample, reason: str) -> RunRecord:
    trace = AttemptTrace(
        attempt_id=1,
        answer="",
        score=0,
        reason=reason,
        token_estimate=0,
        latency_ms=0,
    )
    return RunRecord(
        qid=example.qid,
        question=example.question,
        gold_answer=example.gold_answer,
        agent_type=agent_type,
        predicted_answer="",
        is_correct=False,
        attempts=1,
        token_estimate=0,
        latency_ms=0,
        failure_mode="looping",
        reflections=[],
        traces=[trace],
    )


def run_with_progress(agent: ReActAgent | ReflexionAgent, examples: list[QAExample], label: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    failed = 0
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"{label} (starting)", total=len(examples))
        for example in examples:
            try:
                records.append(agent.run(example))
            except Exception as exc:
                failed += 1
                records.append(_error_record(agent.agent_type, example, f"runtime_error: {exc}"))
            progress.update(task, advance=1, description=f"{label} [{example.qid}]")
    if failed:
        print(f"[yellow]{label}: {failed} item(s) failed and were recorded as incorrect in report.[/yellow]")
    return records

@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "real",
    provider: str = os.getenv("LLM_PROVIDER", "openai"),
    model: str = os.getenv("AGENT_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini"),
    base_url: str = os.getenv("NVIDIA_BASE_URL") or os.getenv("LLM_BASE_URL", ""),
    api_key: str = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", ""),
    timeout_s: int = 120,
    max_retries: int = 2,
    show_progress: bool = True,
) -> None:
    examples = load_dataset(dataset)
    runtime = RuntimeConfig(
        mode="real" if mode.lower() == "real" else "mock",
        provider=provider.lower(),
        model=model,
        base_url=base_url or None,
        api_key=api_key or None,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    react = ReActAgent(runtime=runtime)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, runtime=runtime)

    print(f"[blue]Running[/blue] mode={runtime.mode}, provider={runtime.provider}, model={runtime.model}")
    print(f"[blue]Dataset[/blue] {dataset} ({len(examples)} examples)")

    if show_progress:
        react_records = run_with_progress(react, examples, "ReAct")
        reflexion_records = run_with_progress(reflexion, examples, "Reflexion")
    else:
        react_records = [react.run(example) for example in examples]
        reflexion_records = [reflexion.run(example) for example in examples]

    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=runtime.mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
