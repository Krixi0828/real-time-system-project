# real-time-system-project

RTSPJT 虛擬電廠即時排程專案。系統會產生 periodic task set，建立 72 小時排程，執行 sporadic job acceptance test，安排 aperiodic jobs，並輸出評估結果。

## Environment

- Language: Python 3
- External packages: none
- Run from project root is recommended.

## Folder Structure

```text
.
├── input
│   ├── processor_settings.json
│   └── price_72hr.json
├── output
│   ├── task_set.json
│   ├── schedule_result.json
│   ├── acceptance_test_log.json
│   ├── evaluation_results.json
│   ├── advanced_schedule_result.json
│   ├── advanced_acceptance_test_log.json
│   ├── advanced_evaluation_results.json
│   └── advanced_modeling_notes.json
└── src
    ├── task_generator.py
    ├── scheduler.py
    ├── evaluator.py
    └── advanced_scheduler.py
```

## Level 1 Programs

`src/task_generator.py`

- Generates `output/task_set.json`.
- Periodic job count is computed by actual releases:
  `r, r+p, r+2p, ... <= 72`.
- Validates periodic task constraints including task count, workload density, parameter ranges, non-preemptive count, and frame-size feasibility.

`src/scheduler.py`

- Reads `input/processor_settings.json`, `input/price_72hr.json`, and `output/task_set.json`.
- Schedules jobs in this order:
  1. periodic jobs
  2. sporadic jobs with acceptance test
  3. aperiodic jobs with soft-deadline waiting queue
- Writes:
  - `output/schedule_result.json`
  - `output/acceptance_test_log.json`

`src/evaluator.py`

- Independently evaluates Level 1 outputs.
- Reads schedule and acceptance log from `output/`.
- Writes `output/evaluation_results.json`.

## Level 1 Reproduction

Run:

```bash
python3 src/task_generator.py
python3 src/scheduler.py
python3 src/evaluator.py
```

Expected current result:

```text
constraint_violation_count = 0
hard_deadline_miss_rate = 0.0
soft_deadline_miss_rate = 0.0
sporadic_value_rate = 1.0
```

## Level 2 Program

`src/advanced_scheduler.py`

This program preserves the Level 1 hard-deadline job placement, then performs rolling corrective energy dispatch with relaxed assumptions.

Relaxed assumptions:

- Actual renewable output may differ from forecast.
- Renewable output is constrained by actual availability.
- Battery charging has efficiency loss.
- Battery discharging has efficiency loss.
- SOC updates include charge/discharge efficiency.
- Battery throughput creates aging cost.
- Battery daily throughput is limited.
- Energy dispatch is corrected every 3 hours.
- Renewable surplus charges batteries before market sale.
- Battery discharge is market-aware in high-price hours.
- Aperiodic jobs remain soft-deadline jobs and record miss/tardiness when late.

Run:

```bash
python3 src/advanced_scheduler.py
```

Level 2 outputs:

- `output/advanced_schedule_result.json`
- `output/advanced_acceptance_test_log.json`
- `output/advanced_evaluation_results.json`
- `output/advanced_modeling_notes.json`

Expected current Level 2 result:

```text
advanced_constraint_violation_count = 0
hard_deadline_miss_rate = 0.0
soft_deadline_miss_rate = 0.0
sporadic_value_rate = 1.0
objective_value = 87746.057156
```

## Output File Notes

`schedule_result.json` / `advanced_schedule_result.json`

- `t`: time index, 1 to 72
- `P`: processor output by generator, renewable, and storage
- `k`: job energy allocation from processors
- `sell`: market sale amount
- `soc`: storage state of charge
- `missed_aperiodic`: soft-deadline miss records
- `rejected_sporadic`: sporadic jobs rejected by acceptance test

`evaluation_results.json` / `advanced_evaluation_results.json`

- hard deadline miss rate
- soft deadline miss rate
- average and max tardiness
- average and max response time
- completion-time jitter
- acceptance-test metrics
- sporadic value rate
- generator cost
- market revenue
- objective value

`advanced_modeling_notes.json`

- Level 2 notation
- relaxed assumptions
- text constraints
- mathematical formulas

## Run All Checks

```bash
python3 -m py_compile src/task_generator.py src/scheduler.py src/evaluator.py src/advanced_scheduler.py
python3 src/task_generator.py
python3 src/scheduler.py
python3 src/evaluator.py
python3 src/advanced_scheduler.py
```
