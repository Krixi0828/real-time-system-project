# real-time-system-project

## Environment

- Language: Python 3
- External packages: none

## Level 1 Reproduction

Run these commands from the project root:

```bash
python3 src/task_generator.py
python3 src/scheduler.py
python3 src/evaluator.py
```

Level 1 inputs:

- `input/processor_settings.json`
- `input/price_72hr.json`

Level 1 outputs:

- `output/task_set.json`
- `output/schedule_result.json`
- `output/acceptance_test_log.json`
- `output/evaluation_results.json`

## Level 2 Advanced Scheduler

Run:

```bash
python3 src/advanced_scheduler.py
```

Level 2 relaxed assumptions implemented in `src/advanced_scheduler.py`:

- Renewable actual output may differ from forecast.
- Renewable output is constrained by actual availability.
- Battery charging has efficiency loss.
- Battery discharging has efficiency loss.
- SOC updates include charge/discharge efficiency.
- Battery throughput creates aging cost.
- Battery daily throughput is limited.
- Energy dispatch is corrected every 3 hours.
- Renewable surplus charges batteries before market sale.
- Battery discharge is market-aware in high-price hours.

Level 2 outputs:

- `output/advanced_schedule_result.json`
- `output/advanced_evaluation_results.json`
- `output/advanced_modeling_notes.json`
- `output/advanced_acceptance_test_log.json`

The advanced scheduler preserves Level 1 hard-deadline job placement and redistributes energy supply with actual PV observations, battery charge/discharge, and rolling corrective dispatch.
