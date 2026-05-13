import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

H = 72
FRAME_SIZE = 3
THERMAL_PRIMARY = "thermal_1"
SPORADIC_RESERVE_MWH = 20.0
APERIODIC_RESERVE_MWH = 5.0
ALPHA_MISS_PENALTY = 10000.0

# ============================================================
# JSON utilities + project-root aware path handling
# ============================================================

def find_project_root() -> Path:
    """
    找到專案根目錄，而不是依賴 terminal 目前在哪個資料夾。

    支援兩種執行方式：
    1. 在專案根目錄執行：python3 src/scheduler.py
    2. 進入 src 後執行：python3 scheduler.py

    兩種情況都會讀：
    - PROJECT_ROOT/input/*.json
    - PROJECT_ROOT/output/*.json
    """
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    # 從 scheduler.py 所在位置與目前工作目錄往上找。
    # 用 dict.fromkeys 去重，避免重複檢查同一路徑。
    candidates = list(dict.fromkeys(
        [script_dir, *script_dir.parents, cwd, *cwd.parents]
    ))

    # 標準作業根目錄：同時有 input、output、src。
    for base in candidates:
        if (base / "input").is_dir() and (base / "output").is_dir() and (base / "src").is_dir():
            return base

    # 寬鬆 fallback：至少有 input、output。
    for base in candidates:
        if (base / "input").is_dir() and (base / "output").is_dir():
            return base

    # 最後 fallback：如果此檔案在 src 裡，專案根目錄就是上一層。
    if script_dir.name == "src":
        return script_dir.parent
    return script_dir


PROJECT_ROOT = find_project_root()
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"


def resolve_path(path: str | Path) -> Path:
    """
    將相對路徑轉為 PROJECT_ROOT 下的絕對路徑。

    例如在 src/ 中執行時：
    - input/processor_settings.json
      不會被解讀成 src/input/processor_settings.json
      而是 PROJECT_ROOT/input/processor_settings.json
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: str | Path) -> Any:
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def first_existing(paths: List[str | Path]) -> Path:
    """
    從候選路徑中找第一個存在的檔案。
    所有相對路徑都會先用 resolve_path() 轉到 PROJECT_ROOT 底下。
    """
    checked: List[str] = []
    for p in paths:
        path = resolve_path(p)
        checked.append(str(path))
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot find any of these files:\n- " + "\n- ".join(checked)
    )


def optional_existing(paths: List[str | Path]) -> Optional[Path]:
    """
    找可有可無的檔案，例如 demo_jobs.json。
    找不到時回傳 None，不丟錯。
    """
    for p in paths:
        path = resolve_path(p)
        if path.exists():
            return path
    return None


# ============================================================
# Input loading
# ============================================================

def load_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load required Level-1 inputs.

    The script first follows the submission folder convention:
      input/processor_settings.json
      input/price_72hr.json
      output/task_set.json

    It also supports running directly beside the JSON files during development.
    """
    print(f"Project root: {PROJECT_ROOT}")
    processor_path = first_existing(["input/processor_settings.json", "processor_settings.json"])
    price_path = first_existing(["input/price_72hr.json", "price_72hr.json"])
    task_path = first_existing(["output/task_set.json", "task_set.json"])

    processor_data = load_json(processor_path)
    price_data = load_json(price_path)
    task_data = load_json(task_path)

    print(f"Loaded processor settings from {processor_path}")
    print(f"Loaded price data from {price_path}")
    print(f"Loaded task set from {task_path}")
    return processor_data, price_data, task_data


def load_demo_jobs() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load demo sporadic / aperiodic jobs if provided; otherwise use deterministic samples.

    所有路徑都會以 PROJECT_ROOT 為基準，所以可從 src/ 或專案根目錄執行。
    """
    demo_path = optional_existing([
        "input/demo_jobs.json",
        "output/demo_jobs.json",
        "demo_jobs.json",
    ])
    if demo_path is not None:
        data = load_json(demo_path)
        return data.get("sporadic", []), data.get("aperiodic", [])

    sporadic: Optional[List[Dict[str, Any]]] = None
    aperiodic: Optional[List[Dict[str, Any]]] = None

    sporadic_path = optional_existing([
        "input/sporadic_jobs.json",
        "output/sporadic_jobs.json",
        "sporadic_jobs.json",
    ])
    if sporadic_path is not None:
        data = load_json(sporadic_path)
        sporadic = data.get("sporadic", data) if isinstance(data, dict) else data

    aperiodic_path = optional_existing([
        "input/aperiodic_jobs.json",
        "output/aperiodic_jobs.json",
        "aperiodic_jobs.json",
    ])
    if aperiodic_path is not None:
        data = load_json(aperiodic_path)
        aperiodic = data.get("aperiodic", data) if isinstance(data, dict) else data

    if sporadic is None:
        # 4~7 jobs; e=1~3; w=5~20. Deterministic so the output is reproducible.
        sporadic = [
            {"job_id": "s1", "r": 10, "e": 2, "d": 5, "w": 12, "preempt": 1},
            {"job_id": "s2", "r": 18, "e": 3, "d": 6, "w": 18, "preempt": 0},
            {"job_id": "s3", "r": 29, "e": 1, "d": 4, "w": 20, "preempt": 1},
            {"job_id": "s4", "r": 46, "e": 2, "d": 5, "w": 10, "preempt": 1},
            {"job_id": "s5", "r": 64, "e": 2, "d": 4, "w": 16, "preempt": 0},
        ]

    if aperiodic is None:
        # 7~13 jobs; e=1~4; w=5~15. Soft-deadline jobs.
        aperiodic = [
            {"job_id": "a1", "r": 7, "e": 2, "d": 8, "w": 8, "preempt": 1},
            {"job_id": "a2", "r": 12, "e": 1, "d": 5, "w": 10, "preempt": 1},
            {"job_id": "a3", "r": 21, "e": 4, "d": 10, "w": 15, "preempt": 0},
            {"job_id": "a4", "r": 25, "e": 2, "d": 6, "w": 9, "preempt": 1},
            {"job_id": "a5", "r": 37, "e": 3, "d": 9, "w": 12, "preempt": 0},
            {"job_id": "a6", "r": 52, "e": 1, "d": 4, "w": 5, "preempt": 1},
            {"job_id": "a7", "r": 58, "e": 3, "d": 8, "w": 11, "preempt": 1},
            {"job_id": "a8", "r": 66, "e": 2, "d": 5, "w": 13, "preempt": 0},
        ]

    return normalize_demo_jobs(sporadic, "s"), normalize_demo_jobs(aperiodic, "a")


def normalize_demo_jobs(jobs: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    normalized = []
    for idx, raw in enumerate(jobs, start=1):
        j = dict(raw)
        j.setdefault("job_id", f"{prefix}{idx}")
        # allow either r/d/p style or release/deadline style from demo data
        if "release" in j and "r" not in j:
            j["r"] = j["release"]
        if "execution_time" in j and "e" not in j:
            j["e"] = j["execution_time"]
        if "energy_demand" in j and "w" not in j:
            j["w"] = j["energy_demand"]
        if "relative_deadline" in j and "d" not in j:
            j["d"] = j["relative_deadline"]
        j.setdefault("preempt", 1)
        normalized.append(j)
    return normalized


# ============================================================
# Data maps and validation
# ============================================================

def build_processor_maps(processor_data: Dict[str, Any], price_data: Dict[str, Any]) -> Dict[str, Any]:
    generators = {g["generator_id"]: g for g in processor_data["generator"]}
    storages = {s["storage_id"]: s for s in processor_data["storage"]}
    renewable_capacity = {r["renewable_id"]: r["capacity"] for r in processor_data["renewable_capacity"]}

    renewable_forecast: Dict[str, Dict[int, float]] = {}
    for item in processor_data["renewable_forecast"]:
        for rid, values in item.items():
            renewable_forecast[rid] = {v["hour"]: float(v["pv_forecast"]) for v in values}

    prices = {p["hour"]: float(p["market_price"]) for p in price_data["price"]}
    charging_jobs = {job["job_id"]: job["target_storage"] for job in processor_data["charging_jobs"]}

    return {
        "generators": generators,
        "storages": storages,
        "renewable_capacity": renewable_capacity,
        "renewable_forecast": renewable_forecast,
        "prices": prices,
        "charging_jobs": charging_jobs,
    }


def validate_periodic_task_set(periodic_tasks: Dict[str, Dict[str, Any]], frame_size: int = FRAME_SIZE) -> Dict[str, Any]:
    n = len(periodic_tasks)
    total_jobs = sum(len(release_times(t["r"], t["p"], H)) for t in periodic_tasks.values())
    density = sum(t["e"] / t["p"] for t in periodic_tasks.values())
    periods = sorted({t["p"] for t in periodic_tasks.values()})
    max_e = max(t["e"] for t in periodic_tasks.values())

    violations = []
    if not (6 <= n <= 10):
        violations.append("periodic task count must be 6~10")
    if total_jobs <= 30:
        violations.append("expanded periodic job count must be greater than 30")
    if len(periods) < 3:
        violations.append("at least 3 distinct period values are required")
    if not (0.7 <= density <= 1.0):
        violations.append("periodic workload density must be between 0.7 and 1.0")
    if frame_size < max_e:
        violations.append("frame size must be >= max execution time")
    if H % frame_size != 0:
        violations.append("frame size must divide H=72")

    for task_id, t in periodic_tasks.items():
        for key in ["r", "p", "e", "d", "w", "preempt"]:
            if key not in t:
                violations.append(f"{task_id} missing key {key}")
        if not (1 <= t["r"] <= t["p"]):
            violations.append(f"{task_id}: r must satisfy 1 <= r <= p")
        if not (6 <= t["p"] <= 24):
            violations.append(f"{task_id}: p must be 6~24")
        if not (1 <= t["e"] <= 4):
            violations.append(f"{task_id}: e must be 1~4")
        if not (t["e"] <= t["d"] <= t["p"]):
            violations.append(f"{task_id}: d must satisfy e <= d <= p")
        if not (6 <= t["w"] <= 18):
            violations.append(f"{task_id}: w must be 6~18")
        if 2 * frame_size - math.gcd(frame_size, t["p"]) > t["d"]:
            violations.append(f"{task_id}: frame-size condition fails")

    e_values = [t["e"] for t in periodic_tasks.values()]
    w_values = [t["w"] for t in periodic_tasks.values()]
    if e_values.count(2) < 2:
        violations.append("at least two periodic tasks must have e=2")
    if sum(1 for e in e_values if e >= 3) < 1:
        violations.append("at least one periodic task must have e>=3")
    if sum(1 for w in w_values if w >= 14) < 2:
        violations.append("at least two periodic tasks must have w>=14")
    if sum(1 for t in periodic_tasks.values() if t["d"] == t["e"]) < math.ceil(n * 0.2):
        violations.append("at least 20% of periodic tasks must satisfy d=e")
    if sum(1 for t in periodic_tasks.values() if t["e"] != 1 and t["preempt"] == 0) < 2:
        violations.append("at least two non-preemptive periodic tasks with e!=1 are required")

    if violations:
        raise ValueError("Invalid periodic task set:\n- " + "\n- ".join(violations))

    return {
        "periodic_task_count": n,
        "expanded_periodic_job_count": total_jobs,
        "periodic_workload_density": density,
        "distinct_periods": periods,
        "selected_frame_size": frame_size,
    }


def release_times(r: int, p: int, horizon: int) -> List[int]:
    out = []
    cur = r
    while cur <= horizon:
        out.append(cur)
        cur += p
    return out


# ============================================================
# Job expansion and schedule table
# ============================================================

def expand_periodic_jobs(periodic_tasks: Dict[str, Dict[str, Any]], horizon: int = H) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for task_id, task in periodic_tasks.items():
        for instance, r in enumerate(release_times(task["r"], task["p"], horizon), start=1):
            abs_deadline = r + task["d"] - 1
            if abs_deadline > horizon:
                # The model cannot output a job beyond t=72, so skip incomplete-tail instances.
                continue
            jobs.append({
                "job_id": f"{task_id}_{instance}",
                "task_id": task_id,
                "job_type": "periodic",
                "release": r,
                "relative_deadline": task["d"],
                "deadline": abs_deadline,
                "execution_time": task["e"],
                "energy_demand": float(task["w"]),
                "preempt": int(task["preempt"]),
                "scheduled_times": [],
                "accepted": True,
            })
    jobs.sort(key=lambda j: (j["deadline"], j["release"], j["job_id"]))
    return jobs


def convert_arrival_job(raw: Dict[str, Any], job_type: str) -> Dict[str, Any]:
    r = int(raw["r"])
    e = int(raw["e"])
    d = int(raw["d"])
    return {
        "job_id": str(raw["job_id"]),
        "task_id": str(raw["job_id"]),
        "job_type": job_type,
        "release": r,
        "relative_deadline": d,
        "deadline": min(H, r + d - 1),
        "execution_time": e,
        "energy_demand": float(raw["w"]),
        "preempt": int(raw.get("preempt", 1)),
        "scheduled_times": [],
        "accepted": job_type != "sporadic",
    }


def init_schedule(processor_data: Dict[str, Any], horizon: int = H) -> Dict[int, Dict[str, Any]]:
    all_devices = []
    for g in processor_data["generator"]:
        all_devices.append(g["generator_id"])
    for r in processor_data["renewable_capacity"]:
        all_devices.append(r["renewable_id"])
    for s in processor_data["storage"]:
        all_devices.append(s["storage_id"])

    init_soc = {s["storage_id"]: float(s["soc_init"]) for s in processor_data["storage"]}
    return {
        t: {
            "t": t,
            "P": {device: 0.0 for device in all_devices},
            "k": {},
            "sell": 0.0,
            "soc": init_soc.copy(),
            "missed_aperiodic": [],
            "rejected_sporadic": [],
        }
        for t in range(1, horizon + 1)
    }


# ============================================================
# Time placement: fixed periodic first; sporadic insertion later
# ============================================================

def current_external_load(schedule: Dict[int, Dict[str, Any]], t: int) -> float:
    total = 0.0
    for alloc in schedule[t]["k"].values():
        # charging jobs are not external loads in this scheduler, but this loop is generic.
        total += sum(float(v) for v in alloc.values())
    return total


def candidate_times(job: Dict[str, Any], schedule: Dict[int, Dict[str, Any]], prices: Dict[int, float], max_load: float | Dict[int, float], reserve_after: float) -> Optional[List[int]]:
    """Find times for a job without moving already scheduled jobs.

    Preference order:
      1. Feasible before deadline.
      2. Low market-price hours first, because consuming energy during high-price hours reduces sell revenue.
      3. Earlier completion as tie-breaker for short response time.
    """
    r, dline, e, w = job["release"], job["deadline"], job["execution_time"], job["energy_demand"]
    if r > H or r > dline:
        return None

    def feasible(t: int) -> bool:
        cap = max_load.get(t, 0.0) if isinstance(max_load, dict) else max_load
        return current_external_load(schedule, t) + w <= cap - reserve_after + 1e-9

    if job["preempt"] == 0:
        latest_start = dline - e + 1
        blocks: List[Tuple[float, int, List[int]]] = []
        for start in range(r, latest_start + 1):
            times = list(range(start, start + e))
            if all(1 <= tt <= H and feasible(tt) for tt in times):
                # Prefer blocks that stay inside one frame; this respects the chosen f=3 static-frame idea.
                frame_penalty = 0 if frame_id(times[0]) == frame_id(times[-1]) else 1000
                avg_price = sum(prices.get(tt, 0.0) for tt in times) / len(times)
                blocks.append((frame_penalty + avg_price, times[-1], times))
        if not blocks:
            return None
        blocks.sort(key=lambda x: (x[0], x[1]))
        return blocks[0][2]

    feasible_times = [t for t in range(r, dline + 1) if feasible(t)]
    if len(feasible_times) < e:
        return None
    feasible_times.sort(key=lambda t: (prices.get(t, 0.0), frame_id(t), t))
    chosen = sorted(feasible_times[:e])
    return chosen


def frame_id(t: int, frame_size: int = FRAME_SIZE) -> int:
    return (t - 1) // frame_size + 1


def place_job_energy(schedule: Dict[int, Dict[str, Any]], job: Dict[str, Any], times: List[int], provider: str = THERMAL_PRIMARY) -> None:
    job["scheduled_times"] = sorted(times)
    for t in job["scheduled_times"]:
        jid = job["job_id"]
        if jid not in schedule[t]["k"]:
            schedule[t]["k"][jid] = {}
        schedule[t]["k"][jid][provider] = round(schedule[t]["k"][jid].get(provider, 0.0) + job["energy_demand"], 6)


def schedule_periodic_jobs(schedule: Dict[int, Dict[str, Any]], periodic_jobs: List[Dict[str, Any]], prices: Dict[int, float], max_load: float) -> None:
    for job in periodic_jobs:
        # reserve is used to leave room for future sporadic jobs while constructing the day-ahead schedule.
        times = candidate_times(job, schedule, prices, max_load=max_load, reserve_after=SPORADIC_RESERVE_MWH)
        if times is None:
            # Fallback: still keep the periodic job feasible before deadline, because periodic jobs are mandatory.
            times = candidate_times(job, schedule, prices, max_load=max_load, reserve_after=0.0)
        if times is None:
            raise RuntimeError(f"Cannot schedule mandatory periodic job {job['job_id']} within its deadline")
        place_job_energy(schedule, job, times)


def acceptance_test_and_insert_sporadic(schedule: Dict[int, Dict[str, Any]], sporadic_jobs: List[Dict[str, Any]], prices: Dict[int, float], max_load: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted_jobs: List[Dict[str, Any]] = []
    log: List[Dict[str, Any]] = []

    for raw in sorted(sporadic_jobs, key=lambda x: (x["r"], x["job_id"])):
        job = convert_arrival_job(raw, "sporadic")
        times = candidate_times(job, schedule, prices, max_load=max_load, reserve_after=0.0)

        if times is None:
            for t in range(job["release"], min(job["deadline"], H) + 1):
                schedule[t]["rejected_sporadic"].append(job["job_id"])
            job["accepted"] = False
            reason = "Reject: no feasible time slot with enough remaining energy capacity before hard deadline without moving periodic or already accepted hard-deadline jobs."
            decision = "reject"
        else:
            place_job_energy(schedule, job, times)
            job["accepted"] = True
            accepted_jobs.append(job)
            reason = "Accept: feasible time slot(s) found before hard deadline; insertion does not move existing hard-deadline jobs and keeps energy capacity constraints feasible."
            decision = "accept"

        log.append({
            "job_id": job["job_id"],
            "type": "sporadic",
            "release": job["release"],
            "deadline": job["deadline"],
            "execution_time": job["execution_time"],
            "energy_demand": job["energy_demand"],
            "preempt": job["preempt"],
            "decision": decision,
            "scheduled_times": job["scheduled_times"],
            "reason": reason,
        })

    return accepted_jobs, log


def schedule_aperiodic_waiting_queue(schedule: Dict[int, Dict[str, Any]], aperiodic_jobs: List[Dict[str, Any]], prices: Dict[int, float], max_load: float) -> List[Dict[str, Any]]:
    scheduled: List[Dict[str, Any]] = []
    waiting = [convert_arrival_job(raw, "aperiodic") for raw in sorted(aperiodic_jobs, key=lambda x: (x["r"], x["job_id"]))]

    for job in waiting:
        # First try to finish before soft deadline while preserving a small emergency reserve.
        times = candidate_times(job, schedule, prices, max_load=max_load, reserve_after=APERIODIC_RESERVE_MWH)
        if times is None:
            # Soft-deadline fallback: finish by H even if tardy.
            late_job = dict(job)
            late_job["deadline"] = H
            times = candidate_times(late_job, schedule, prices, max_load=max_load, reserve_after=0.0)
        if times is None:
            # This should rarely happen with the current constant 80 MWh supply strategy.
            job["scheduled_times"] = []
        else:
            place_job_energy(schedule, job, times)

        if job["scheduled_times"] and max(job["scheduled_times"]) > job["deadline"]:
            schedule[max(job["scheduled_times"])]["missed_aperiodic"].append(job["job_id"])
        elif not job["scheduled_times"]:
            schedule[H]["missed_aperiodic"].append(job["job_id"])
        scheduled.append(job)

    return scheduled


# ============================================================
# Energy allocation: constant thermal reserve strategy
# ============================================================

def planned_generator_outputs(t: int, maps: Dict[str, Any]) -> Dict[str, float]:
    """A ramp-feasible conservative unit-commitment plan.

    Both thermal units start from initial_energy=0. Therefore, they cannot jump
    directly to maximum output at t=1. The plan ramps each unit up according to
    its ramp_up_rate until reaching output_max, then keeps it online.
    """
    out = {}
    for gid, g in maps["generators"].items():
        max_p = float(g["output_max"])
        ru = float(g["ramp_up_rate"])
        # The first positive hour must also respect output_min; given the input
        # satisfies output_min <= ramp_up_rate, ru is a feasible first output.
        out[gid] = min(max_p, ru * t)
    return out


def build_capacity_by_hour(maps: Dict[str, Any]) -> Dict[int, float]:
    return {t: sum(planned_generator_outputs(t, maps).values()) for t in range(1, H + 1)}


def finalize_energy_balance(schedule: Dict[int, Dict[str, Any]], maps: Dict[str, Any]) -> None:
    """Set processor outputs and sell values.

    Level-1 strategy:
      - Use a ramp-feasible conservative plan for all thermal generators.
      - The unused amount is sold to the market.
      - Renewables and batteries remain at 0 output; batteries keep their initial SOC.
    """
    generators = maps["generators"]
    storages = maps["storages"]

    for t in range(1, H + 1):
        demand = current_external_load(schedule, t)
        planned = planned_generator_outputs(t, maps)
        total_output = sum(planned.values())
        if demand > total_output + 1e-9:
            raise RuntimeError(f"Hour {t}: demand {demand} exceeds planned thermal output {total_output}")

        # Reset all P to zero, then apply planned thermal outputs.
        for dev in schedule[t]["P"]:
            schedule[t]["P"][dev] = 0.0
        for gid, p in planned.items():
            schedule[t]["P"][gid] = round(p, 6)

        # Re-distribute each job's k over online thermal generators so that
        # constraint 20 holds for each generator.
        remaining = {gid: float(schedule[t]["P"][gid]) for gid in generators}
        for jid in sorted(schedule[t]["k"]):
            need = sum(float(v) for v in schedule[t]["k"][jid].values())
            new_alloc = {}
            for gid in sorted(generators):
                take = min(need, remaining[gid])
                if take > 1e-9:
                    new_alloc[gid] = round(take, 6)
                    remaining[gid] -= take
                    need -= take
                if need <= 1e-9:
                    break
            if need > 1e-6:
                raise RuntimeError(f"Hour {t}: cannot allocate enough generator energy to {jid}")
            schedule[t]["k"][jid] = new_alloc

        # SOC stays constant because this Level-1 schedule does not charge/discharge storage.
        for sid, s in storages.items():
            schedule[t]["soc"][sid] = float(s["soc_init"])

        schedule[t]["sell"] = round(total_output - demand, 6)
        if schedule[t]["sell"] < -1e-9:
            raise RuntimeError(f"Hour {t}: negative sell value after balancing")


def validate_full_schedule(schedule: Dict[int, Dict[str, Any]], maps: Dict[str, Any], all_jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    violations: List[str] = []
    generators = maps["generators"]
    storages = maps["storages"]
    renewable_capacity = maps["renewable_capacity"]
    renewable_forecast = maps["renewable_forecast"]

    # Job timing and energy completeness.
    for job in all_jobs:
        if job["job_type"] == "sporadic" and not job.get("accepted", False):
            continue
        times = job.get("scheduled_times", [])
        if len(times) != job["execution_time"]:
            violations.append(f"{job['job_id']} execution time mismatch")
        if job["job_type"] in ["periodic", "sporadic"] and times and max(times) > job["deadline"]:
            violations.append(f"{job['job_id']} misses hard deadline")
        if any(t < job["release"] for t in times):
            violations.append(f"{job['job_id']} scheduled before release")
        if job["preempt"] == 0 and times:
            st = sorted(times)
            if any(st[i] != st[i - 1] + 1 for i in range(1, len(st))):
                violations.append(f"{job['job_id']} is non-preemptive but not continuous")
        for t in times:
            supplied = sum(schedule[t]["k"].get(job["job_id"], {}).values())
            if abs(supplied - job["energy_demand"]) > 1e-6:
                violations.append(f"{job['job_id']} energy demand not fully supplied at t={t}")

    # Processor constraints and energy balance.
    prev_p = {gid: float(g.get("initial_energy", 0.0)) for gid, g in generators.items()}
    for t in range(1, H + 1):
        row = schedule[t]

        for gid, g in generators.items():
            p = float(row["P"].get(gid, 0.0))
            if p > 0:
                if p < float(g["output_min"]) - 1e-9 or p > float(g["output_max"]) + 1e-9:
                    violations.append(f"{gid} output bound violation at t={t}")
            if p - prev_p[gid] > float(g["ramp_up_rate"]) + 1e-9:
                violations.append(f"{gid} ramp-up violation at t={t}")
            if prev_p[gid] - p > float(g["ramp_down_rate"]) + 1e-9:
                violations.append(f"{gid} ramp-down violation at t={t}")
            prev_p[gid] = p

        for rid, cap in renewable_capacity.items():
            p = float(row["P"].get(rid, 0.0))
            available = float(cap) * float(renewable_forecast[rid].get(t, 0.0))
            if p < -1e-9 or p > available + 1e-9:
                violations.append(f"{rid} renewable forecast violation at t={t}")

        for sid, s in storages.items():
            p = float(row["P"].get(sid, 0.0))
            soc = float(row["soc"].get(sid, 0.0))
            if p < -1e-9 or p > float(s["discharge_max"]) + 1e-9:
                violations.append(f"{sid} discharge bound violation at t={t}")
            if soc < float(s["soc_min"]) - 1e-9 or soc > float(s["soc_max"]) + 1e-9:
                violations.append(f"{sid} SOC bound violation at t={t}")

        if float(row["sell"]) < -1e-9:
            violations.append(f"negative sell at t={t}")

        total_p = sum(float(v) for v in row["P"].values())
        total_k = sum(sum(float(v) for v in alloc.values()) for alloc in row["k"].values())
        if abs(total_p - total_k - float(row["sell"])) > 1e-6:
            violations.append(f"energy balance violation at t={t}")

    if violations:
        raise RuntimeError("Schedule validation failed:\n- " + "\n- ".join(violations[:30]))
    return {"constraint_violation_count": 0, "checked_hours": H, "checked_jobs": len(all_jobs)}


# ============================================================
# Evaluation metrics
# ============================================================

def completion_time(job: Dict[str, Any]) -> Optional[int]:
    return max(job["scheduled_times"]) if job.get("scheduled_times") else None


def response_time(job: Dict[str, Any]) -> Optional[int]:
    c = completion_time(job)
    return None if c is None else c - job["release"] + 1


def evaluate(schedule: Dict[int, Dict[str, Any]], periodic_jobs: List[Dict[str, Any]], sporadic_jobs: List[Dict[str, Any]], aperiodic_jobs: List[Dict[str, Any]], raw_sporadic: List[Dict[str, Any]], maps: Dict[str, Any]) -> Dict[str, Any]:
    hard_jobs = periodic_jobs + [j for j in sporadic_jobs if j.get("accepted", False)]
    hard_misses = [j for j in hard_jobs if completion_time(j) is None or completion_time(j) > j["deadline"]]

    soft_misses = [j for j in aperiodic_jobs if completion_time(j) is None or completion_time(j) > j["deadline"]]

    all_scheduled_jobs = periodic_jobs + sporadic_jobs + aperiodic_jobs
    tardiness_values = []
    response_values = []
    for j in all_scheduled_jobs:
        if j["job_type"] == "sporadic" and not j.get("accepted", False):
            continue
        c = completion_time(j)
        if c is None:
            tardiness_values.append(H - j["deadline"] + 1)
            continue
        tardiness_values.append(max(0, c - j["deadline"]))
        response_values.append(c - j["release"] + 1)

    # Completion-time jitter: average absolute difference between consecutive completion times
    # for instances generated by the same periodic task.
    jitter_values = []
    by_task: Dict[str, List[int]] = {}
    for j in periodic_jobs:
        c = completion_time(j)
        if c is not None:
            by_task.setdefault(j["task_id"], []).append(c)
    for vals in by_task.values():
        vals.sort()
        for i in range(1, len(vals)):
            jitter_values.append(abs(vals[i] - vals[i - 1]))

    total_sporadic_e = sum(int(j["e"]) for j in raw_sporadic) or 1
    completed_sporadic_e = sum(j["execution_time"] for j in sporadic_jobs if j.get("accepted") and completion_time(j) is not None and completion_time(j) <= j["deadline"])

    generator_cost = 0.0
    for t in range(1, H + 1):
        for gid, g in maps["generators"].items():
            p = float(schedule[t]["P"].get(gid, 0.0))
            if p > 0:
                generator_cost += float(g["cost_fixed"]) + float(g["cost_variable"]) * p

    market_revenue = sum(maps["prices"].get(t, 0.0) * float(schedule[t]["sell"]) for t in range(1, H + 1))
    objective_value = ALPHA_MISS_PENALTY * len(soft_misses) + generator_cost - market_revenue

    return {
        "hard_deadline_miss_rate": round(len(hard_misses) / len(hard_jobs), 6) if hard_jobs else 0.0,
        "soft_deadline_miss_rate": round(len(soft_misses) / len(aperiodic_jobs), 6) if aperiodic_jobs else 0.0,
        "average_tardiness": round(sum(tardiness_values) / len(tardiness_values), 6) if tardiness_values else 0.0,
        "max_tardiness": max(tardiness_values) if tardiness_values else 0,
        "average_response_time": round(sum(response_values) / len(response_values), 6) if response_values else 0.0,
        "max_response_time": max(response_values) if response_values else 0,
        "completion_time_jitter": round(sum(jitter_values) / len(jitter_values), 6) if jitter_values else 0.0,
        "acceptance_test": {
            "sporadic_total_jobs": len(raw_sporadic),
            "sporadic_accepted_jobs": sum(1 for j in sporadic_jobs if j.get("accepted")),
            "sporadic_rejected_jobs": sum(1 for j in sporadic_jobs if not j.get("accepted")),
            "post_acceptance_violation_rate": 0.0,
        },
        "sporadic_value_rate": round(completed_sporadic_e / total_sporadic_e, 6),
        "generator_cost": round(generator_cost, 6),
        "market_revenue": round(market_revenue, 6),
        "objective_value": round(objective_value, 6),
        "periodic_average_response_time": round(sum(response_time(j) or 0 for j in periodic_jobs) / len(periodic_jobs), 6),
        "periodic_max_response_time": max(response_time(j) or 0 for j in periodic_jobs),
        "soft_missed_jobs": [j["job_id"] for j in soft_misses],
        "hard_missed_jobs": [j["job_id"] for j in hard_misses],
    }


def export_schedule(schedule: Dict[int, Dict[str, Any]], output_path: str = "output/schedule_result.json") -> None:
    save_json({"schedule_result": [schedule[t] for t in range(1, H + 1)]}, output_path)


# ============================================================
# Main flow
# ============================================================

def main() -> None:
    processor_data, price_data, task_data = load_inputs()
    maps = build_processor_maps(processor_data, price_data)

    periodic_tasks = task_data["periodic"]
    task_set_summary = validate_periodic_task_set(periodic_tasks, FRAME_SIZE)
    print("Periodic task set summary:", task_set_summary)

    capacity_by_hour = build_capacity_by_hour(maps)
    schedule = init_schedule(processor_data, H)

    periodic_jobs = expand_periodic_jobs(periodic_tasks, H)
    schedule_periodic_jobs(schedule, periodic_jobs, maps["prices"], capacity_by_hour)

    raw_sporadic, raw_aperiodic = load_demo_jobs()
    accepted_sporadic, acceptance_log = acceptance_test_and_insert_sporadic(schedule, raw_sporadic, maps["prices"], capacity_by_hour)
    scheduled_aperiodic = schedule_aperiodic_waiting_queue(schedule, raw_aperiodic, maps["prices"], capacity_by_hour)

    finalize_energy_balance(schedule, maps)
    validation_summary = validate_full_schedule(schedule, maps, periodic_jobs + accepted_sporadic + scheduled_aperiodic)

    evaluation = evaluate(schedule, periodic_jobs, accepted_sporadic, scheduled_aperiodic, raw_sporadic, maps)
    evaluation["task_set_summary"] = task_set_summary
    evaluation["validation_summary"] = validation_summary
    evaluation["reserve_strategy"] = {
        "selected_frame_size": FRAME_SIZE,
        "day_ahead_sporadic_reserve_mwh": SPORADIC_RESERVE_MWH,
        "aperiodic_reserve_mwh": APERIODIC_RESERVE_MWH,
        "primary_generator": THERMAL_PRIMARY,
        "planned_thermal_capacity_by_hour": capacity_by_hour,
        "strategy": "Periodic jobs are fixed first; sporadic jobs are inserted only into remaining capacity without moving existing hard-deadline jobs; aperiodic jobs wait in a queue and use remaining slack."
    }

    export_schedule(schedule, "output/schedule_result.json")
    save_json(evaluation, "output/evaluation_results.json")
    save_json({"acceptance_test_log": acceptance_log}, "output/acceptance_test_log.json")

    print("Done.")
    print("Wrote output/schedule_result.json")
    print("Wrote output/evaluation_results.json")
    print("Wrote output/acceptance_test_log.json")


if __name__ == "__main__":
    main()
