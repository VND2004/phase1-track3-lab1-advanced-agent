# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: real
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.39 | 0.46 | 0.07 |
| Avg attempts | 1 | 1.85 | 0.85 |
| Avg token estimate | 4310.18 | 7335.96 | 3025.78 |
| Avg latency (ms) | 6315.57 | 15374.92 | 9059.35 |

## Failure modes
```json
{
  "react": {
    "none": 39,
    "wrong_final_answer": 60,
    "looping": 1
  },
  "reflexion": {
    "none": 46,
    "wrong_final_answer": 30,
    "looping": 24
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
