"""
evaluator.py
============================================================
RTSPJT Level 1 independent evaluator（獨立評估器）

這支程式的角色不是重新排程，而是「檢查 scheduler.py 產生的結果」。

主要輸入：
- input/processor_settings.json
- input/price_72hr.json
- output/task_set.json
- output/schedule_result.json
- output/acceptance_test_log.json
- input/demo_jobs.json（若有 Demo 提供 sporadic / aperiodic jobs）

主要輸出：
- output/evaluation_results.json

對應 Level 1 評分標準：
1. Periodic Task Set 設計：檢查 1-1 ~ 1-8
2. 模型限制式：檢查 constraints 1~23 的主要可驗證項目
3. 排程結果與 Periodic Task 效能：檢查 periodic jobs 是否完成、deadline、response time
4. Acceptance Test：統計 sporadic accept/reject、sporadic value rate、post-acceptance violation
5. 評估指標：計算 miss rate、tardiness、response time、jitter、cost、revenue、objective
6. 日前保留策略效能分析：保留 scheduler 輸出的 reserve_strategy，輔助報告撰寫
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

H = 72
FRAME_SIZE = 3
ALPHA_MISS_PENALTY = 10000
EPS = 1e-6


# ============================================================
# 1. JSON 讀寫與路徑處理工具
# ============================================================

def load_json(path: str | Path) -> Any:
    """
    讀取 JSON 檔案。

    完成作業哪部分：
    - 支援繳交格式中的 input/*.json 與 output/*.json。
    - 確保 evaluator 可以獨立讀取 scheduler 的輸出結果。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """
    儲存 JSON 檔案。

    完成作業哪部分：
    - 產生 output/evaluation_results.json。
    - 對應 Level 1 評分標準第 5 大項「評估指標」。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def first_existing_path(candidates: List[str]) -> Path:
    """
    從多個候選路徑中找到第一個存在的檔案。

    設計原因：
    - 你可能在專案根目錄執行，也可能在 src/ 裡執行。
    - 因此 evaluator 同時支援 input/...、output/... 與同層檔案。
    """
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find any of these files: {candidates}")


# ============================================================
# 2. 載入輸入資料
# ============================================================

def load_all_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    載入 evaluator 需要的所有資料。

    讀取：
    - processor_settings.json：發電機、PV、battery 參數
    - price_72hr.json：72 小時市場價格
    - task_set.json：periodic task set
    - schedule_result.json：scheduler 產生的排程結果
    - acceptance_test_log.json：sporadic acceptance test 紀錄

    完成作業哪部分：
    - 對應繳交資料夾中的必要 input/output files。
    - evaluator 會用這些資料重新計算 evaluation_results.json。
    """
    processor_path = first_existing_path([
        "input/processor_settings.json",
        "../input/processor_settings.json",
        "processor_settings.json",
    ])
    price_path = first_existing_path([
        "input/price_72hr.json",
        "../input/price_72hr.json",
        "price_72hr.json",
    ])
    task_path = first_existing_path([
        "output/task_set.json",
        "../output/task_set.json",
        "task_set.json",
    ])
    schedule_path = first_existing_path([
        "output/schedule_result.json",
        "../output/schedule_result.json",
        "schedule_result.json",
    ])

    # acceptance_test_log.json 在沒有 sporadic jobs 時也可能不存在，因此這裡允許 fallback 成空 log。
    try:
        acceptance_path = first_existing_path([
            "output/acceptance_test_log.json",
            "../output/acceptance_test_log.json",
            "acceptance_test_log.json",
        ])
        acceptance_data = load_json(acceptance_path)
    except FileNotFoundError:
        acceptance_data = {"acceptance_test_log": []}

    return (
        load_json(processor_path),
        load_json(price_path),
        load_json(task_path),
        load_json(schedule_path),
        acceptance_data,
    )


# ============================================================
# 3. 建立 processor / price 查詢表
# ============================================================

def build_maps(processor_data: Dict[str, Any], price_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    將 processor_settings.json 與 price_72hr.json 轉成方便查詢的 dict。

    會建立：
    - generators：thermal generator 參數
    - storages：battery 參數
    - renewable_capacity：PV 額定容量
    - renewable_forecast：PV 每小時 forecast
    - prices：每小時 market price
    - charging_jobs：charging job 對應的 battery
    - device_sets：所有設備集合、generator 集合、PV 集合、battery 集合

    完成作業哪部分：
    - 對應 Level 1 模型限制式中的傳統機組、再生能源、儲能設備與市場價格。
    - 後續 constraints 6~23 都會用到這些查詢表。
    """
    generators = {g["generator_id"]: g for g in processor_data.get("generator", [])}
    storages = {s["storage_id"]: s for s in processor_data.get("storage", [])}

    renewable_capacity = {
        r["renewable_id"]: float(r["capacity"])
        for r in processor_data.get("renewable_capacity", [])
    }

    renewable_forecast: Dict[str, Dict[int, float]] = {}
    for item in processor_data.get("renewable_forecast", []):
        for rid, values in item.items():
            renewable_forecast[rid] = {int(v["hour"]): float(v["pv_forecast"]) for v in values}

    prices = {int(p["hour"]): float(p["market_price"]) for p in price_data.get("price", [])}

    charging_jobs = {
        job["job_id"]: job["target_storage"]
        for job in processor_data.get("charging_jobs", [])
    }

    generator_ids = set(generators.keys())
    renewable_ids = set(renewable_capacity.keys())
    storage_ids = set(storages.keys())
    all_devices = generator_ids | renewable_ids | storage_ids

    return {
        "generators": generators,
        "storages": storages,
        "renewable_capacity": renewable_capacity,
        "renewable_forecast": renewable_forecast,
        "prices": prices,
        "charging_jobs": charging_jobs,
        "generator_ids": generator_ids,
        "renewable_ids": renewable_ids,
        "storage_ids": storage_ids,
        "all_devices": all_devices,
    }


# ============================================================
# 4. 載入 Demo jobs：sporadic / aperiodic
# ============================================================

def normalize_demo_jobs(jobs: List[Dict[str, Any]], prefix: str, job_type: str) -> List[Dict[str, Any]]:
    """
    將 Demo job 欄位標準化。

    支援欄位：
    - job_id / id
    - r / release / release_time
    - e / execution_time
    - d / deadline / relative_deadline
    - w / energy_demand
    - preempt / preemptable / preemptive

    注意：
    - input 中的 d 通常是 relative deadline。
    - evaluator 會另外建立 absolute_deadline = r + d - 1。

    完成作業哪部分：
    - 支援 Level 1 demo 時提供的 sporadic / aperiodic jobs。
    - Sporadic jobs 用來計算 acceptance test 與 sporadic value rate。
    - Aperiodic jobs 用來計算 soft deadline miss rate 與 tardiness。
    """
    normalized = []
    for idx, raw in enumerate(jobs, start=1):
        job_id = raw.get("job_id", raw.get("id", f"{prefix}{idx}"))
        r = int(raw.get("r", raw.get("release", raw.get("release_time"))))
        e = int(raw.get("e", raw.get("execution_time")))
        d = int(raw.get("d", raw.get("deadline", raw.get("relative_deadline"))))
        w = float(raw.get("w", raw.get("energy_demand")))
        preempt = int(raw.get("preempt", raw.get("preemptable", raw.get("preemptive", 1))))

        normalized.append({
            "job_id": str(job_id),
            "type": job_type,
            "release": r,
            "relative_deadline": d,
            "deadline": r + d - 1,
            "execution_time": e,
            "energy_demand": w,
            "preempt": preempt,
        })
    return normalized


def load_demo_jobs_from_files_or_fallback(acceptance_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    載入 Demo jobs。

    優先順序：
    1. input/demo_jobs.json 或 output/demo_jobs.json
    2. input/sporadic_jobs.json + input/aperiodic_jobs.json
    3. acceptance_test_log.json 中的 sporadic jobs
    4. 若仍沒有資料，使用與 scheduler.py 相同的 deterministic sample jobs

    完成作業哪部分：
    - 對應 Level 1 的 sporadic / aperiodic demo jobs。
    - 讓 evaluator 可以在 demo jobs 出現後重新計算 metrics。
    """
    demo_candidates = [
        "input/demo_jobs.json", "../input/demo_jobs.json", "output/demo_jobs.json", "demo_jobs.json"
    ]
    for candidate in demo_candidates:
        path = Path(candidate)
        if path.exists():
            data = load_json(path)
            return (
                normalize_demo_jobs(data.get("sporadic", []), "s", "sporadic"),
                normalize_demo_jobs(data.get("aperiodic", []), "a", "aperiodic"),
            )

    sporadic_raw: Optional[List[Dict[str, Any]]] = None
    aperiodic_raw: Optional[List[Dict[str, Any]]] = None

    for candidate in ["input/sporadic_jobs.json", "../input/sporadic_jobs.json", "output/sporadic_jobs.json", "sporadic_jobs.json"]:
        path = Path(candidate)
        if path.exists():
            data = load_json(path)
            sporadic_raw = data.get("sporadic", data) if isinstance(data, dict) else data
            break

    for candidate in ["input/aperiodic_jobs.json", "../input/aperiodic_jobs.json", "output/aperiodic_jobs.json", "aperiodic_jobs.json"]:
        path = Path(candidate)
        if path.exists():
            data = load_json(path)
            aperiodic_raw = data.get("aperiodic", data) if isinstance(data, dict) else data
            break

    # 如果沒有 sporadic_jobs.json，就從 acceptance_test_log.json 還原 sporadic jobs。
    if sporadic_raw is None:
        logs = acceptance_data.get("acceptance_test_log", [])
        sporadic_raw = []
        for row in logs:
            if row.get("type") == "sporadic":
                # acceptance log 裡的 deadline 已經是 absolute deadline，轉回 relative deadline。
                r = int(row["release"])
                abs_deadline = int(row["deadline"])
                sporadic_raw.append({
                    "job_id": row["job_id"],
                    "r": r,
                    "e": row["execution_time"],
                    "d": abs_deadline - r + 1,
                    "w": row["energy_demand"],
                    "preempt": row.get("preempt", 1),
                })

    # 如果仍沒有任何 demo jobs，使用 scheduler.py 的 deterministic sample。
    if not sporadic_raw:
        sporadic_raw = [
            {"job_id": "s1", "r": 10, "e": 2, "d": 5, "w": 12, "preempt": 1},
            {"job_id": "s2", "r": 18, "e": 3, "d": 6, "w": 18, "preempt": 0},
            {"job_id": "s3", "r": 29, "e": 1, "d": 4, "w": 20, "preempt": 1},
            {"job_id": "s4", "r": 46, "e": 2, "d": 5, "w": 10, "preempt": 1},
            {"job_id": "s5", "r": 64, "e": 2, "d": 4, "w": 16, "preempt": 0},
        ]

    if aperiodic_raw is None:
        aperiodic_raw = [
            {"job_id": "a1", "r": 7, "e": 2, "d": 8, "w": 8, "preempt": 1},
            {"job_id": "a2", "r": 12, "e": 1, "d": 5, "w": 10, "preempt": 1},
            {"job_id": "a3", "r": 21, "e": 4, "d": 10, "w": 15, "preempt": 0},
            {"job_id": "a4", "r": 25, "e": 2, "d": 6, "w": 9, "preempt": 1},
            {"job_id": "a5", "r": 37, "e": 3, "d": 9, "w": 12, "preempt": 0},
            {"job_id": "a6", "r": 52, "e": 1, "d": 4, "w": 5, "preempt": 1},
            {"job_id": "a7", "r": 58, "e": 3, "d": 8, "w": 11, "preempt": 1},
            {"job_id": "a8", "r": 66, "e": 2, "d": 5, "w": 13, "preempt": 0},
        ]

    return (
        normalize_demo_jobs(sporadic_raw, "s", "sporadic"),
        normalize_demo_jobs(aperiodic_raw, "a", "aperiodic"),
    )


# ============================================================
# 5. Periodic task set 檢查與展開
# ============================================================

def validate_periodic_task_set(task_data: Dict[str, Any], frame_size: int = FRAME_SIZE) -> Tuple[Dict[str, Any], List[str]]:
    """
    檢查 periodic task set 是否符合 Level 1 評分標準 1-1 ~ 1-8。

    檢查項目：
    - 1-1：欄位完整 r, p, e, d, w, preempt
    - 1-2：task 數量 6~10
    - 1-3：展開後 periodic jobs 數量 > 30
    - 1-4：參數範圍與至少 3 種 period、至少 2 個 e=2、至少 1 個 e>=3、至少 2 個 w>=14
    - 1-5：workload density >= 0.7
    - 1-6：至少 20% task 滿足 d=e
    - 1-7：至少 2 個 e!=1 且 non-preemptive
    - 1-8：frame size 合法

    回傳：
    - summary：task set 統計資料
    - violations：違反項目清單
    """
    violations: List[str] = []
    tasks = task_data.get("periodic", {})
    num_tasks = len(tasks)

    required_fields = {"r", "p", "e", "d", "w", "preempt"}
    for task_id, task in tasks.items():
        missing = required_fields - set(task.keys())
        if missing:
            violations.append(f"1-1: task {task_id} missing fields {sorted(missing)}")

    if not (6 <= num_tasks <= 10):
        violations.append(f"1-2: periodic task count must be 6~10, got {num_tasks}")

    expanded_count = count_expanded_periodic_jobs(tasks)
    if expanded_count <= 30:
        violations.append(f"1-3: expanded periodic jobs must be > 30, got {expanded_count}")

    periods = [int(t["p"]) for t in tasks.values()]
    executions = [int(t["e"]) for t in tasks.values()]
    deadlines = [int(t["d"]) for t in tasks.values()]
    weights = [float(t["w"]) for t in tasks.values()]

    for task_id, t in tasks.items():
        r, p, e, d, w = int(t["r"]), int(t["p"]), int(t["e"]), int(t["d"]), float(t["w"])
        if not (1 <= r <= p):
            violations.append(f"1-4: {task_id} violates 1 <= r <= p")
        if not (6 <= p <= 24):
            violations.append(f"1-4: {task_id} violates 6 <= p <= 24")
        if not (1 <= e <= 4):
            violations.append(f"1-4: {task_id} violates 1 <= e <= 4")
        if not (e <= d <= p):
            violations.append(f"1-4: {task_id} violates e <= d <= p")
        if not (6 <= w <= 18):
            violations.append(f"1-4: {task_id} violates 6 <= w <= 18")

    if len(set(periods)) < 3:
        violations.append("1-4: at least 3 distinct period values are required")
    if executions.count(2) < 2:
        violations.append("1-4: at least 2 periodic tasks must have e = 2")
    if sum(1 for e in executions if e >= 3) < 1:
        violations.append("1-4: at least 1 periodic task must have e >= 3")
    if sum(1 for w in weights if w >= 14) < 2:
        violations.append("1-4: at least 2 periodic tasks must have w >= 14")

    workload_density = sum(int(t["e"]) / int(t["p"]) for t in tasks.values()) if tasks else 0.0
    if workload_density < 0.7:
        violations.append(f"1-5: workload density must be >= 0.7, got {workload_density:.4f}")

    required_de_count = math.ceil(num_tasks * 0.2)
    de_count = sum(1 for t in tasks.values() if int(t["d"]) == int(t["e"]))
    if de_count < required_de_count:
        violations.append(f"1-6: at least 20% tasks must satisfy d=e, got {de_count}/{num_tasks}")

    non_preemptive_count = sum(1 for t in tasks.values() if int(t["e"]) != 1 and int(t["preempt"]) == 0)
    if non_preemptive_count < 2:
        violations.append("1-7: at least 2 tasks with e!=1 must be non-preemptive")

    max_e = max(executions) if executions else 0
    if frame_size < max_e:
        violations.append(f"1-8: frame size f must be >= max(e), got f={frame_size}, max(e)={max_e}")
    if H % frame_size != 0:
        violations.append(f"1-8: H mod f must be 0, got H={H}, f={frame_size}")
    for task_id, t in tasks.items():
        p = int(t["p"])
        d = int(t["d"])
        if 2 * frame_size - math.gcd(frame_size, p) > d:
            violations.append(
                f"1-8: {task_id} violates 2f - gcd(f,p) <= d "
                f"with f={frame_size}, p={p}, d={d}"
            )

    summary = {
        "periodic_task_count": num_tasks,
        "expanded_periodic_job_count": expanded_count,
        "periodic_workload_density": workload_density,
        "distinct_periods": sorted(set(periods)),
        "selected_frame_size": frame_size,
        "deadline_equals_execution_count": de_count,
        "non_preemptive_e_gt_1_count": non_preemptive_count,
        "high_energy_task_count_w_ge_14": sum(1 for w in weights if w >= 14),
    }
    return summary, violations


def count_expanded_periodic_jobs(periodic_tasks: Dict[str, Dict[str, Any]], horizon: int = H) -> int:
    """
    計算展開後的 periodic jobs 數量。

    完成作業哪部分：
    - 對應評分標準 1-3：展開後 periodic jobs 數量必須大於 30。
    """
    return len(expand_periodic_jobs(periodic_tasks, horizon))


def expand_periodic_jobs(periodic_tasks: Dict[str, Dict[str, Any]], horizon: int = H) -> List[Dict[str, Any]]:
    """
    將 periodic task template 展開成 periodic job instances。

    每個 task：
    - release = r, r+p, r+2p, ...
    - absolute deadline = release + d - 1
    - 若 deadline > 72，表示該 job 無法完整在 72 小時範圍內檢查，這裡不納入。

    完成作業哪部分：
    - Step 3：展開 periodic tasks 成 periodic jobs。
    - 評分標準 3-2、3-3：檢查 periodic jobs 是否完整執行並準時完成。
    """
    jobs: List[Dict[str, Any]] = []
    for task_id, t in periodic_tasks.items():
        release = int(t["r"])
        period = int(t["p"])
        instance = 1
        while release <= horizon:
            deadline = release + int(t["d"]) - 1
            if deadline <= horizon:
                jobs.append({
                    "job_id": f"{task_id}_{instance}",
                    "task_id": task_id,
                    "type": "periodic",
                    "release": release,
                    "relative_deadline": int(t["d"]),
                    "deadline": deadline,
                    "execution_time": int(t["e"]),
                    "energy_demand": float(t["w"]),
                    "preempt": int(t["preempt"]),
                })
            release += period
            instance += 1
    return jobs


# ============================================================
# 6. 從 schedule_result.json 還原 job 執行狀態
# ============================================================

def normalize_schedule(schedule_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    將 schedule_result.json 轉成 schedule[t] 的格式。

    完成作業哪部分：
    - 對應評分標準 3-1：確認 72 單位時間日前固定排程可以被 evaluator 讀取。
    - 後續所有 constraints 檢查都會使用這份 schedule。
    """
    rows = schedule_data.get("schedule_result", [])
    schedule = {int(row["t"]): row for row in rows}
    return schedule


def extract_job_executions(schedule: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    從 schedule_result.json 的 k 欄位還原每個 job 的執行時段與電量。

    對每個 t：
    - schedule[t]["k"][job_id][device] 代表 job_id 在 t 從 device 取得多少電。
    - 若該 job 在 t 的總取得電量 > 0，表示該 job 在 t 有執行。

    完成作業哪部分：
    - Constraint 1：檢查 job 每個執行時段是否完整取得 energy demand。
    - Constraint 2：檢查 job 是否在 release 前執行。
    - Constraint 3 / 4：計算 job 是否在 deadline 前完成。
    - 評估指標：completion time、response time、tardiness。
    """
    executions: Dict[str, Dict[str, Any]] = {}
    for t, row in schedule.items():
        for job_id, allocation in row.get("k", {}).items():
            total_energy = sum(float(v) for v in allocation.values())
            if total_energy <= EPS:
                continue
            if job_id not in executions:
                executions[job_id] = {"times": [], "energy_by_time": {}, "allocation_by_time": {}}
            executions[job_id]["times"].append(t)
            executions[job_id]["energy_by_time"][t] = total_energy
            executions[job_id]["allocation_by_time"][t] = dict(allocation)

    for info in executions.values():
        info["times"].sort()
    return executions


def completion_time(job_id: str, executions: Dict[str, Dict[str, Any]]) -> Optional[int]:
    """
    回傳 job 的 completion time C_j。

    若 job 沒有任何執行時段，回傳 None。

    完成作業哪部分：
    - 評分標準 5-3：tardiness 需要 C_j。
    - 評分標準 5-4：response time 需要 C_j。
    """
    times = executions.get(job_id, {}).get("times", [])
    return max(times) if times else None


# ============================================================
# 7. Job-level constraints 檢查
# ============================================================

def check_job_timing_and_energy(jobs: List[Dict[str, Any]], executions: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    檢查每個 job 的時間限制與每時段 energy demand。

    檢查內容：
    - Constraint 1：只要 job 在 t 執行，該時段總供電量必須等於 w_j。
    - Constraint 2：job 不可在 release time 前執行。
    - Constraint 3：periodic / accepted sporadic hard-deadline jobs 必須在 deadline 前做滿 e_j。
    - Constraint 4：aperiodic jobs 最終應在 H 前做滿 e_j；若晚於 soft deadline，記為 miss。
    - Constraint 5：non-preemptive jobs 必須連續執行。

    完成作業哪部分：
    - 對應 Level 1 評分標準 2-1、2-2。
    - 對應評分標準 3-2、3-3。
    """
    violations: List[str] = []

    for job in jobs:
        job_id = job["job_id"]
        times = executions.get(job_id, {}).get("times", [])
        energy_by_time = executions.get(job_id, {}).get("energy_by_time", {})

        if any(t < job["release"] for t in times):
            violations.append(f"Constraint 2: {job_id} runs before release time")

        if len(times) != int(job["execution_time"]):
            violations.append(
                f"Constraint 3/4: {job_id} execution time mismatch, "
                f"expected {job['execution_time']}, got {len(times)}"
            )

        for t in times:
            supplied = float(energy_by_time[t])
            required = float(job["energy_demand"])
            if abs(supplied - required) > EPS:
                violations.append(
                    f"Constraint 1: {job_id} at t={t} energy mismatch, "
                    f"required {required}, got {supplied}"
                )

        if job["type"] in {"periodic", "sporadic"}:
            c = completion_time(job_id, executions)
            if c is None or c > int(job["deadline"]):
                violations.append(
                    f"Constraint 3: hard-deadline job {job_id} misses deadline, "
                    f"completion={c}, deadline={job['deadline']}"
                )

        if int(job.get("preempt", 1)) == 0 and times:
            sorted_times = sorted(times)
            for idx in range(1, len(sorted_times)):
                if sorted_times[idx] != sorted_times[idx - 1] + 1:
                    violations.append(f"Constraint 5: non-preemptive job {job_id} is not continuous: {sorted_times}")
                    break

    return violations


# ============================================================
# 8. 每小時設備與能源限制式檢查
# ============================================================

def check_hourly_energy_constraints(schedule: Dict[int, Dict[str, Any]], maps: Dict[str, Any]) -> List[str]:
    """
    檢查每小時與設備相關的 constraints。

    檢查內容：
    - Constraints 6~12：傳統機組 output min/max、ramp up/down、min up/down、初始狀態
    - Constraint 13：PV 不可超過 forecast 可用量
    - Constraints 14~19、21：battery 放電、充電、SOC、不可同時充放電、充電來源限制
    - Constraint 20：各設備 P 是否足以供應其 k allocation
    - Constraint 22：sell >= 0
    - Constraint 23：每小時總供給 = job 用電 + battery charging + sell

    完成作業哪部分：
    - 對應 Level 1 評分標準 2-3、2-4、2-5、2-6、2-7。
    """
    violations: List[str] = []

    violations.extend(check_generator_constraints(schedule, maps))
    violations.extend(check_renewable_constraints(schedule, maps))
    violations.extend(check_storage_constraints(schedule, maps))
    violations.extend(check_supply_allocation_and_balance(schedule, maps))

    return violations


def check_generator_constraints(schedule: Dict[int, Dict[str, Any]], maps: Dict[str, Any]) -> List[str]:
    """
    檢查傳統機組 constraints 6~12。

    主要檢查：
    - 若 P_i,t > 0，必須 output_min <= P_i,t <= output_max。
    - 相鄰兩時段不可超過 ramp_up_rate / ramp_down_rate。
    - 開機後需滿足 min_up_time。
    - 關機後需滿足 min_down_time。
    - 初始 on/off time 若尚未滿足，也需補足。
    """
    violations: List[str] = []
    generators = maps["generators"]

    for gid, g in generators.items():
        output_min = float(g["output_min"])
        output_max = float(g["output_max"])
        ru = float(g["ramp_up_rate"])
        rd = float(g["ramp_down_rate"])
        ut = int(g["min_up_time"])
        dt = int(g["min_down_time"])
        initial_p = float(g.get("initial_energy", 0))
        initial_on_time = int(g.get("initial_on_time", 0))
        initial_off_time = int(g.get("initial_off_time", 0))

        p_by_t = {0: initial_p}
        on_by_t = {0: initial_p > EPS}
        for t in range(1, H + 1):
            p = float(schedule[t].get("P", {}).get(gid, 0.0))
            p_by_t[t] = p
            on_by_t[t] = p > EPS

            if p > EPS and not (output_min - EPS <= p <= output_max + EPS):
                violations.append(
                    f"Constraint 6: {gid} output at t={t} must be within [{output_min}, {output_max}], got {p}"
                )

            if p - p_by_t[t - 1] > ru + EPS:
                violations.append(
                    f"Constraint 7: {gid} ramp-up violation at t={t}, prev={p_by_t[t-1]}, now={p}, RU={ru}"
                )
            if p_by_t[t - 1] - p > rd + EPS:
                violations.append(
                    f"Constraint 7: {gid} ramp-down violation at t={t}, prev={p_by_t[t-1]}, now={p}, RD={rd}"
                )

        # Constraint 8 是參數可行性檢查。
        if output_min > ru + EPS:
            violations.append(f"Constraint 8: {gid} output_min > ramp_up_rate")

        # Constraints 9~10：檢查排程內發生的 off->on 與 on->off transition。
        for t in range(1, H + 1):
            turned_on = on_by_t[t] and not on_by_t[t - 1]
            turned_off = (not on_by_t[t]) and on_by_t[t - 1]

            if turned_on and t + ut - 1 <= H:
                if not all(on_by_t[tau] for tau in range(t, t + ut)):
                    violations.append(f"Constraint 9: {gid} violates min_up_time after startup at t={t}")

            if turned_off and t + dt - 1 <= H:
                if any(on_by_t[tau] for tau in range(t, t + dt)):
                    violations.append(f"Constraint 10: {gid} violates min_down_time after shutdown at t={t}")

        # Constraints 11~12：初始狀態尚未滿足 min up/down 時的補足。
        if initial_on_time > 0 and initial_on_time < ut:
            remaining = ut - initial_on_time
            if not all(on_by_t[t] for t in range(1, min(remaining, H) + 1)):
                violations.append(f"Constraint 11: {gid} does not satisfy remaining initial min_up_time")

        if initial_off_time > 0 and initial_off_time < dt:
            remaining = dt - initial_off_time
            if any(on_by_t[t] for t in range(1, min(remaining, H) + 1)):
                violations.append(f"Constraint 12: {gid} does not satisfy remaining initial min_down_time")

    return violations


def check_renewable_constraints(schedule: Dict[int, Dict[str, Any]], maps: Dict[str, Any]) -> List[str]:
    """
    檢查再生能源 constraint 13。

    對每個 PV：
    - 0 <= P_pv,t <= capacity * forecast[t]
    """
    violations: List[str] = []
    for rid in maps["renewable_ids"]:
        capacity = maps["renewable_capacity"][rid]
        forecast = maps["renewable_forecast"].get(rid, {})
        for t in range(1, H + 1):
            p = float(schedule[t].get("P", {}).get(rid, 0.0))
            limit = capacity * float(forecast.get(t, 0.0))
            if p < -EPS or p > limit + EPS:
                violations.append(
                    f"Constraint 13: {rid} at t={t} exceeds renewable forecast, P={p}, limit={limit}"
                )
    return violations


def compute_storage_charge(schedule_row: Dict[str, Any], storage_id: str, maps: Dict[str, Any]) -> float:
    """
    計算某一個 storage 在某一小時的充電量。

    根據作業定義：
    - charging job 例如 battery_1_chg 會把電充進 battery_1。
    - charging job 的供應來源只能是 generator 或 renewable。

    完成作業哪部分：
    - Constraint 15：充電量不可超過 charge_max。
    - Constraint 16：SOC 更新需要 charge amount。
    - Constraint 19：不可同時充電與放電。
    """
    total_charge = 0.0
    for job_id, target_storage in maps["charging_jobs"].items():
        if target_storage != storage_id:
            continue
        allocation = schedule_row.get("k", {}).get(job_id, {})
        total_charge += sum(float(v) for device, v in allocation.items() if device in maps["generator_ids"] | maps["renewable_ids"])
    return total_charge


def check_storage_constraints(schedule: Dict[int, Dict[str, Any]], maps: Dict[str, Any]) -> List[str]:
    """
    檢查儲能設備 constraints 14~19、21。

    主要檢查：
    - Constraint 14：battery 放電 P 不可超過 discharge_max。
    - Constraint 15：battery 充電量不可超過 charge_max。
    - Constraint 16：SOC[t] = SOC[t-1] + charge - discharge。
    - Constraint 17：SOC 需在 soc_min ~ soc_max。
    - Constraint 18：不可放出超過最低存量以下的電。
    - Constraint 19：同一時間不可同時充電與放電。
    - Constraint 21：charging job 不能由 battery 供應。
    """
    violations: List[str] = []

    for bid, storage in maps["storages"].items():
        soc_min = float(storage["soc_min"])
        soc_max = float(storage["soc_max"])
        discharge_max = float(storage["discharge_max"])
        charge_max = float(storage["charge_max"])
        prev_soc = float(storage["soc_init"])

        for t in range(1, H + 1):
            row = schedule[t]
            discharge = float(row.get("P", {}).get(bid, 0.0))
            charge = compute_storage_charge(row, bid, maps)
            reported_soc = float(row.get("soc", {}).get(bid, prev_soc))
            expected_soc = prev_soc + charge - discharge

            if discharge < -EPS or discharge > discharge_max + EPS:
                violations.append(f"Constraint 14: {bid} discharge at t={t} exceeds max, got {discharge}")

            if charge < -EPS or charge > charge_max + EPS:
                violations.append(f"Constraint 15: {bid} charge at t={t} exceeds max, got {charge}")

            if abs(reported_soc - expected_soc) > 1e-4:
                violations.append(
                    f"Constraint 16: {bid} SOC mismatch at t={t}, "
                    f"expected {expected_soc}, reported {reported_soc}"
                )

            if reported_soc < soc_min - EPS or reported_soc > soc_max + EPS:
                violations.append(
                    f"Constraint 17: {bid} SOC at t={t} outside [{soc_min}, {soc_max}], got {reported_soc}"
                )

            if discharge > max(0.0, prev_soc - soc_min) + EPS:
                violations.append(
                    f"Constraint 18: {bid} discharges below minimum SOC at t={t}, "
                    f"prev_soc={prev_soc}, discharge={discharge}, soc_min={soc_min}"
                )

            if discharge > EPS and charge > EPS:
                violations.append(f"Constraint 19: {bid} charges and discharges at the same time t={t}")

            prev_soc = reported_soc

    # Constraint 21：charging job 不可由 battery 供應。
    for t, row in schedule.items():
        for chg_job_id in maps["charging_jobs"]:
            allocation = row.get("k", {}).get(chg_job_id, {})
            for device, amount in allocation.items():
                if device not in maps["generator_ids"] | maps["renewable_ids"] and float(amount) > EPS:
                    violations.append(
                        f"Constraint 21: charging job {chg_job_id} at t={t} is supplied by non-generator/non-renewable device {device}"
                    )

    return violations


def check_supply_allocation_and_balance(schedule: Dict[int, Dict[str, Any]], maps: Dict[str, Any]) -> List[str]:
    """
    檢查設備供電分配與每小時能源平衡。

    檢查：
    - Constraint 20：各設備分配出去的 k 不可超過 P。
      - generator / renewable 可供應外部 job 與 charging job。
      - battery 只能供應外部 job，不能供應 charging job。
    - Constraint 22：sell >= 0。
    - Constraint 23：sum(P) = external job demand + charging demand + sell。
    """
    violations: List[str] = []

    for t, row in schedule.items():
        P = {device: float(value) for device, value in row.get("P", {}).items()}
        k = row.get("k", {})
        sell = float(row.get("sell", 0.0))

        if sell < -EPS:
            violations.append(f"Constraint 22: sell at t={t} is negative, got {sell}")

        device_allocation = {device: 0.0 for device in maps["all_devices"]}
        external_demand = 0.0
        charging_demand = 0.0

        for job_id, allocation in k.items():
            is_charging_job = job_id in maps["charging_jobs"]
            job_total = 0.0
            for device, amount in allocation.items():
                amount = float(amount)
                job_total += amount
                device_allocation[device] = device_allocation.get(device, 0.0) + amount

                if is_charging_job and device in maps["storage_ids"] and amount > EPS:
                    violations.append(
                        f"Constraint 20/21: battery {device} supplies charging job {job_id} at t={t}"
                    )

            if is_charging_job:
                charging_demand += job_total
            else:
                external_demand += job_total

        for device, allocated in device_allocation.items():
            available = P.get(device, 0.0)
            if allocated > available + EPS:
                violations.append(
                    f"Constraint 20: device {device} at t={t} allocated {allocated} but P={available}"
                )

        total_supply = sum(P.values())
        total_demand = external_demand + charging_demand + sell
        if abs(total_supply - total_demand) > 1e-4:
            violations.append(
                f"Constraint 23: energy balance violation at t={t}, "
                f"supply={total_supply}, demand+charge+sell={total_demand}"
            )

    return violations


# ============================================================
# 9. Acceptance test 評估
# ============================================================

def parse_acceptance_log(acceptance_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    將 acceptance_test_log.json 轉成以 job_id 為 key 的 dict。

    完成作業哪部分：
    - 評分標準 4-1：acceptance test 方法與紀錄。
    - 評分標準 4-2：accept / reject 判斷原因。
    - 評分標準 4-3：sporadic value rate。
    """
    logs = acceptance_data.get("acceptance_test_log", [])
    return {row["job_id"]: row for row in logs}


def compute_acceptance_metrics(
    sporadic_jobs: List[Dict[str, Any]],
    acceptance_log: Dict[str, Dict[str, Any]],
    executions: Dict[str, Dict[str, Any]],
    hard_violations: List[str],
) -> Dict[str, Any]:
    """
    計算 sporadic acceptance test 相關指標。

    計算：
    - sporadic_total_jobs
    - sporadic_accepted_jobs
    - sporadic_rejected_jobs
    - sporadic_value_rate
    - post_acceptance_violation_rate

    sporadic_value_rate 定義：
    - 在 hard deadline 前完成的 sporadic execution time 總和 / Demo 提供的 sporadic execution time 總和。

    完成作業哪部分：
    - 對應評分標準 4-3。
    """
    total_e = sum(int(j["execution_time"]) for j in sporadic_jobs)
    completed_e_before_deadline = 0

    accepted = 0
    rejected = 0
    for job in sporadic_jobs:
        job_id = job["job_id"]
        decision = acceptance_log.get(job_id, {}).get("decision", "unknown")
        if decision == "accept":
            accepted += 1
        elif decision == "reject":
            rejected += 1

        times = executions.get(job_id, {}).get("times", [])
        if len(times) == int(job["execution_time"]) and times and max(times) <= int(job["deadline"]):
            completed_e_before_deadline += int(job["execution_time"])

    value_rate = completed_e_before_deadline / total_e if total_e > 0 else 0.0

    # 這裡採用簡單定義：若 accepted sporadic 造成任何 hard violation，視為 post-acceptance violation。
    post_acceptance_violation_rate = 1.0 if hard_violations else 0.0

    return {
        "sporadic_total_jobs": len(sporadic_jobs),
        "sporadic_accepted_jobs": accepted,
        "sporadic_rejected_jobs": rejected,
        "sporadic_unknown_decision_jobs": len(sporadic_jobs) - accepted - rejected,
        "sporadic_value_rate": round(value_rate, 6),
        "post_acceptance_violation_rate": round(post_acceptance_violation_rate, 6),
    }


# ============================================================
# 10. 評估指標計算
# ============================================================

def compute_job_metrics(jobs: List[Dict[str, Any]], executions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    計算所有 jobs 的 completion time、response time、tardiness。

    定義：
    - C_j = completion time
    - R_j = C_j - r_j
    - T_j = max(0, C_j - d_j)

    完成作業哪部分：
    - 評分標準 5-3：Avg / Max Tardiness。
    - 評分標準 5-4：Avg / Max Response Time。
    """
    per_job: Dict[str, Dict[str, Any]] = {}
    response_times: List[int] = []
    tardiness_values: List[int] = []

    for job in jobs:
        job_id = job["job_id"]
        c = completion_time(job_id, executions)
        if c is None:
            # 沒完成時，tardiness 以 H+1 - deadline 作為 conservative penalty。
            response = None
            tardiness = max(0, H + 1 - int(job["deadline"]))
        else:
            response = c - int(job["release"])
            tardiness = max(0, c - int(job["deadline"]))
            response_times.append(response)

        tardiness_values.append(tardiness)
        per_job[job_id] = {
            "type": job["type"],
            "release": job["release"],
            "deadline": job["deadline"],
            "completion_time": c,
            "response_time": response,
            "tardiness": tardiness,
            "scheduled_times": executions.get(job_id, {}).get("times", []),
        }

    return {
        "per_job": per_job,
        "average_response_time": round(sum(response_times) / len(response_times), 6) if response_times else 0.0,
        "max_response_time": max(response_times) if response_times else 0,
        "average_tardiness": round(sum(tardiness_values) / len(tardiness_values), 6) if tardiness_values else 0.0,
        "max_tardiness": max(tardiness_values) if tardiness_values else 0,
    }


def compute_miss_rates(
    periodic_jobs: List[Dict[str, Any]],
    accepted_sporadic_jobs: List[Dict[str, Any]],
    aperiodic_jobs: List[Dict[str, Any]],
    executions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    計算 hard deadline miss rate 與 soft deadline miss rate。

    hard deadline jobs：
    - periodic jobs
    - accepted sporadic jobs

    soft deadline jobs：
    - aperiodic jobs

    注意：
    - rejected sporadic jobs 不視為 hard deadline miss，因為 acceptance test 可以合法拒絕。
    - 但 rejected sporadic 會降低 sporadic value rate。

    完成作業哪部分：
    - 評分標準 5-1：Hard deadline miss rate。
    - 評分標準 5-2：Soft deadline miss rate。
    """
    hard_jobs = periodic_jobs + accepted_sporadic_jobs
    hard_missed = []
    for job in hard_jobs:
        c = completion_time(job["job_id"], executions)
        if c is None or c > int(job["deadline"]):
            hard_missed.append(job["job_id"])

    soft_missed = []
    for job in aperiodic_jobs:
        c = completion_time(job["job_id"], executions)
        if c is None or c > int(job["deadline"]):
            soft_missed.append(job["job_id"])

    return {
        "hard_deadline_miss_rate": round(len(hard_missed) / len(hard_jobs), 6) if hard_jobs else 0.0,
        "soft_deadline_miss_rate": round(len(soft_missed) / len(aperiodic_jobs), 6) if aperiodic_jobs else 0.0,
        "hard_missed_jobs": hard_missed,
        "soft_missed_jobs": soft_missed,
    }


def compute_completion_time_jitter(periodic_jobs: List[Dict[str, Any]], executions: Dict[str, Dict[str, Any]]) -> float:
    """
    計算 periodic jobs 的 completion-time jitter。

    這裡採用的定義：
    - 對同一個 periodic task 的不同 instances，先計算 completion offset：C_j - release_j。
    - 再計算該 task 的 offset 標準差。
    - 最後對所有 task 的標準差取平均。

    為什麼這樣做：
    - 作業只說要對同一 periodic task 的不同 job instances 計算 completion-time jitter，
      沒有指定公式，因此這裡用標準差作為 jitter 指標。

    完成作業哪部分：
    - 評分標準 5-5：Completion-time Jitter。
    """
    by_task: Dict[str, List[int]] = {}
    for job in periodic_jobs:
        c = completion_time(job["job_id"], executions)
        if c is None:
            continue
        offset = c - int(job["release"])
        by_task.setdefault(job["task_id"], []).append(offset)

    jitters = []
    for offsets in by_task.values():
        if len(offsets) >= 2:
            jitters.append(statistics.pstdev(offsets))
        else:
            jitters.append(0.0)

    return round(sum(jitters) / len(jitters), 6) if jitters else 0.0


# ============================================================
# 11. 成本、收益、objective value
# ============================================================

def compute_cost_revenue_objective(
    schedule: Dict[int, Dict[str, Any]],
    maps: Dict[str, Any],
    soft_missed_count: int,
) -> Dict[str, float]:
    """
    計算 generator cost、market revenue 與 objective value。

    Objective：
    F = alpha * f1 + f2 + f3
    - f1 = aperiodic miss 數量
    - f2 = 傳統機組發電成本
    - f3 = -market revenue

    成本：
    - 若 generator P_i,t > 0，支付 fixed cost。
    - variable cost = cost_variable * P_i,t。

    收益：
    - revenue = sell_t * market_price_t。

    完成作業哪部分：
    - 評分標準 5：generator_cost、market_revenue、objective_value。
    - 評分標準 6-2：報告中的 objective trade-off 分析可引用這些數據。
    """
    generator_cost = 0.0
    market_revenue = 0.0

    for t in range(1, H + 1):
        row = schedule[t]
        for gid, g in maps["generators"].items():
            p = float(row.get("P", {}).get(gid, 0.0))
            if p > EPS:
                generator_cost += float(g["cost_fixed"]) + float(g["cost_variable"]) * p

        sell = float(row.get("sell", 0.0))
        market_revenue += sell * float(maps["prices"].get(t, 0.0))

    objective_value = ALPHA_MISS_PENALTY * soft_missed_count + generator_cost - market_revenue

    return {
        "generator_cost": round(generator_cost, 6),
        "market_revenue": round(market_revenue, 6),
        "objective_value": round(objective_value, 6),
    }


# ============================================================
# 12. 綜合執行 evaluator
# ============================================================

def run_evaluation() -> Dict[str, Any]:
    """
    evaluator 主流程。

    流程：
    1. 讀取 processor_settings、price、task_set、schedule_result、acceptance_test_log。
    2. 檢查 periodic task set 是否符合評分標準 1-1~1-8。
    3. 展開 periodic jobs。
    4. 載入 demo sporadic / aperiodic jobs。
    5. 從 schedule_result.json 還原每個 job 的執行時段。
    6. 檢查 job timing、energy demand、non-preemptive constraints。
    7. 檢查 generator / renewable / battery / sell / energy balance constraints。
    8. 計算 miss rate、response time、tardiness、jitter、acceptance metrics。
    9. 計算 generator cost、market revenue、objective value。
    10. 輸出 evaluation_results.json。

    完成作業哪部分：
    - 這支程式整體對應 Level 1 第 5 大項「評估指標」。
    - 同時協助檢查第 1、2、3、4、6 大項是否合理。
    """
    processor_data, price_data, task_data, schedule_data, acceptance_data = load_all_inputs()
    maps = build_maps(processor_data, price_data)
    schedule = normalize_schedule(schedule_data)
    acceptance_log = parse_acceptance_log(acceptance_data)

    # 檢查 schedule 是否有 72 小時。
    format_violations: List[str] = []
    if len(schedule) != H:
        format_violations.append(f"Schedule result must contain {H} hours, got {len(schedule)}")
    for t in range(1, H + 1):
        if t not in schedule:
            format_violations.append(f"Schedule result missing hour t={t}")

    task_set_summary, task_set_violations = validate_periodic_task_set(task_data, FRAME_SIZE)
    periodic_jobs = expand_periodic_jobs(task_data.get("periodic", {}), H)

    raw_sporadic_jobs, aperiodic_jobs = load_demo_jobs_from_files_or_fallback(acceptance_data)

    # accepted sporadic jobs 才是 hard-deadline schedule 中必須完成的 jobs。
    accepted_sporadic_jobs = [
        job for job in raw_sporadic_jobs
        if acceptance_log.get(job["job_id"], {}).get("decision") == "accept"
    ]

    all_jobs_for_constraint_check = periodic_jobs + accepted_sporadic_jobs + aperiodic_jobs
    executions = extract_job_executions(schedule)

    job_violations = check_job_timing_and_energy(all_jobs_for_constraint_check, executions)
    hourly_violations = check_hourly_energy_constraints(schedule, maps)
    all_violations = format_violations + task_set_violations + job_violations + hourly_violations

    # Hard violation 不包含 task set 文字規格，而是實際排程與系統限制式違反。
    hard_system_violations = job_violations + hourly_violations + format_violations

    miss_info = compute_miss_rates(periodic_jobs, accepted_sporadic_jobs, aperiodic_jobs, executions)
    all_scheduled_jobs = periodic_jobs + accepted_sporadic_jobs + aperiodic_jobs
    metric_info = compute_job_metrics(all_scheduled_jobs, executions)
    periodic_metric_info = compute_job_metrics(periodic_jobs, executions)
    acceptance_metrics = compute_acceptance_metrics(raw_sporadic_jobs, acceptance_log, executions, hard_system_violations)
    cost_info = compute_cost_revenue_objective(schedule, maps, len(miss_info["soft_missed_jobs"]))
    jitter = compute_completion_time_jitter(periodic_jobs, executions)

    evaluation = {
        # Level 1 評分標準 5-1 ~ 5-5
        "hard_deadline_miss_rate": miss_info["hard_deadline_miss_rate"],
        "soft_deadline_miss_rate": miss_info["soft_deadline_miss_rate"],
        "average_tardiness": metric_info["average_tardiness"],
        "max_tardiness": metric_info["max_tardiness"],
        "average_response_time": metric_info["average_response_time"],
        "max_response_time": metric_info["max_response_time"],
        "completion_time_jitter": jitter,

        # Level 1 評分標準 4：Acceptance Test
        "acceptance_test": {
            "sporadic_total_jobs": acceptance_metrics["sporadic_total_jobs"],
            "sporadic_accepted_jobs": acceptance_metrics["sporadic_accepted_jobs"],
            "sporadic_rejected_jobs": acceptance_metrics["sporadic_rejected_jobs"],
            "sporadic_unknown_decision_jobs": acceptance_metrics["sporadic_unknown_decision_jobs"],
            "post_acceptance_violation_rate": acceptance_metrics["post_acceptance_violation_rate"],
        },
        "sporadic_value_rate": acceptance_metrics["sporadic_value_rate"],

        # Objective 相關指標
        **cost_info,

        # Periodic jobs 專用效能，對應評分標準 3-3。
        "periodic_average_response_time": periodic_metric_info["average_response_time"],
        "periodic_max_response_time": periodic_metric_info["max_response_time"],

        # Miss list
        "soft_missed_jobs": miss_info["soft_missed_jobs"],
        "hard_missed_jobs": miss_info["hard_missed_jobs"],

        # Task set summary，對應評分標準 1。
        "task_set_summary": task_set_summary,

        # Validation summary，對應評分標準 2。
        "validation_summary": {
            "constraint_violation_count": len(all_violations),
            "task_set_violation_count": len(task_set_violations),
            "format_violation_count": len(format_violations),
            "job_constraint_violation_count": len(job_violations),
            "hourly_constraint_violation_count": len(hourly_violations),
            "checked_hours": len(schedule),
            "checked_jobs": len(all_scheduled_jobs),
        },

        # 詳細違規清單：方便 debug 與報告說明。
        "violations": all_violations,

        # 每個 job 的細節：方便你確認 response time / tardiness 怎麼算。
        "per_job_metrics": metric_info["per_job"],
    }

    # 如果 scheduler 原本的 evaluation_results.json 裡有 reserve_strategy，就保留下來。
    # 這可以讓 report 寫第 6 大項「日前保留策略效能分析」時有資料可引用。
    try:
        old_eval_path = first_existing_path([
            "output/evaluation_results.json",
            "../output/evaluation_results.json",
            "evaluation_results.json",
        ])
        old_eval = load_json(old_eval_path)
        if "reserve_strategy" in old_eval:
            evaluation["reserve_strategy"] = old_eval["reserve_strategy"]
    except FileNotFoundError:
        pass

    return evaluation


def main() -> None:
    """
    程式進入點。

    執行方式：
    python3 evaluator.py

    輸出：
    output/evaluation_results.json
    """
    evaluation = run_evaluation()
    save_json(evaluation, "output/evaluation_results.json")

    print("Evaluation finished.")
    print(f"constraint_violation_count = {evaluation['validation_summary']['constraint_violation_count']}")
    print(f"hard_deadline_miss_rate = {evaluation['hard_deadline_miss_rate']}")
    print(f"soft_deadline_miss_rate = {evaluation['soft_deadline_miss_rate']}")
    print(f"sporadic_value_rate = {evaluation['sporadic_value_rate']}")
    print("Saved to output/evaluation_results.json")


if __name__ == "__main__":
    main()
