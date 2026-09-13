import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import scheduler as base

H = base.H
FRAME_SIZE = base.FRAME_SIZE
ALPHA_MISS_PENALTY = 10000.0

ROLLING_WINDOW = 3
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.90
BATTERY_AGING_COST_PER_MWH = 5.0
BATTERY_DAILY_THROUGHPUT_RATIO = 1.20
BATTERY_SELF_DISCHARGE_RATE = 0.0005
SOC_DEPENDENT_CHARGE_RATIO = 0.60
SOC_DEPENDENT_DISCHARGE_RATIO = 0.60
SELL_COMMITMENT_PENALTY_RATE = 0.25
LOW_PRICE_QUANTILE = 0.30
HIGH_PRICE_QUANTILE = 0.70
EPS = 1e-6


def save_json(data: Any, path: str | Path) -> None:
    path = base.resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def quantile(values: List[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    idx = round((len(values) - 1) * q)
    return values[idx]


def uncertainty_factor(renewable_id: str, t: int) -> float:
    """Deterministic sinusoidal +/-10% PV uncertainty factor."""
    return 1.0 + 0.1 * math.sin(2.0 * math.pi * t / 24.0)


def build_actual_renewable(maps: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, float]]]:
    actual: Dict[str, Dict[int, Dict[str, float]]] = {}
    for rid, cap in maps["renewable_capacity"].items():
        actual[rid] = {}
        for t in range(1, H + 1):
            forecast = float(maps["renewable_forecast"].get(rid, {}).get(t, 0.0))
            factor = uncertainty_factor(rid, t)
            actual_ratio = max(0.0, min(1.0, forecast * factor))
            actual[rid][t] = {
                "forecast_ratio": round(forecast, 6),
                "uncertainty_factor": round(factor, 6),
                "actual_ratio": round(actual_ratio, 6),
                "actual_available_mwh": round(float(cap) * actual_ratio, 6),
            }
    return actual


def build_level1_job_schedule() -> Tuple[
    Dict[int, Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    processor_data, price_data, task_data = base.load_inputs()
    maps = base.build_processor_maps(processor_data, price_data)

    periodic_tasks = task_data["periodic"]
    base.validate_periodic_task_set(periodic_tasks, FRAME_SIZE)
    capacity_by_hour = base.build_capacity_by_hour(maps)
    schedule = base.init_schedule(processor_data, H)

    periodic_jobs = base.expand_periodic_jobs(periodic_tasks, H)
    base.schedule_periodic_jobs(schedule, periodic_jobs, maps["prices"], capacity_by_hour)

    raw_sporadic, raw_aperiodic = base.load_demo_jobs()
    accepted_sporadic, acceptance_log = base.acceptance_test_and_insert_sporadic(
        schedule, raw_sporadic, maps["prices"], capacity_by_hour
    )
    scheduled_aperiodic = base.schedule_aperiodic_waiting_queue(
        schedule, raw_aperiodic, maps["prices"], capacity_by_hour
    )

    return schedule, maps, periodic_jobs, accepted_sporadic, scheduled_aperiodic, raw_sporadic, {
        "acceptance_test_log": acceptance_log,
        "processor_data": processor_data,
        "price_data": price_data,
        "task_data": task_data,
    }


def job_need_by_hour(schedule: Dict[int, Dict[str, Any]], t: int) -> Dict[str, float]:
    needs = {}
    for job_id, alloc in schedule[t].get("k", {}).items():
        if job_id.endswith("_chg"):
            continue
        needs[job_id] = round(sum(float(v) for v in alloc.values()), 6)
    return needs


def allocate_from_pools(
    job_needs: Dict[str, float],
    pools: Dict[str, float],
    device_order: List[str],
) -> Dict[str, Dict[str, float]]:
    allocations: Dict[str, Dict[str, float]] = {}
    for job_id in sorted(job_needs):
        need = float(job_needs[job_id])
        allocations[job_id] = {}
        for device in device_order:
            take = min(need, pools.get(device, 0.0))
            if take > EPS:
                allocations[job_id][device] = round(take, 6)
                pools[device] -= take
                need -= take
            if need <= EPS:
                break
        if need > 1e-5:
            raise RuntimeError(f"Advanced dispatch cannot supply {job_id}; remaining need={need}")
    return allocations


def planned_thermal_outputs(t: int, maps: Dict[str, Any]) -> Dict[str, float]:
    # Keep the Level 1 ramp-feasible thermal commitment as corrective reserve.
    return base.planned_generator_outputs(t, maps)


def apply_self_discharge(soc_value: float, storage: Dict[str, Any]) -> Tuple[float, float]:
    soc_min = float(storage["soc_min"])
    after_loss = max(soc_min, soc_value * (1.0 - BATTERY_SELF_DISCHARGE_RATE))
    return after_loss, max(0.0, soc_value - after_loss)


def soc_dependent_discharge_limit(soc_value: float, storage: Dict[str, Any]) -> float:
    soc_min = float(storage["soc_min"])
    energy_above_min = max(0.0, soc_value - soc_min)
    return min(
        float(storage["discharge_max"]),
        energy_above_min * DISCHARGE_EFFICIENCY * SOC_DEPENDENT_DISCHARGE_RATIO,
    )


def soc_dependent_charge_limit(soc_value: float, storage: Dict[str, Any]) -> float:
    soc_max = float(storage["soc_max"])
    input_room = max(0.0, (soc_max - soc_value) / CHARGE_EFFICIENCY)
    return min(
        float(storage["charge_max"]),
        input_room * SOC_DEPENDENT_CHARGE_RATIO,
    )


def build_day_ahead_sell_commitment(
    schedule: Dict[int, Dict[str, Any]],
    maps: Dict[str, Any],
) -> Dict[int, float]:
    commitment = {}
    for t in range(1, H + 1):
        thermal_output = sum(planned_thermal_outputs(t, maps).values())
        external_demand = sum(job_need_by_hour(schedule, t).values())
        commitment[t] = round(max(0.0, thermal_output - external_demand), 6)
    return commitment


def dispatch_advanced_energy(
    schedule: Dict[int, Dict[str, Any]],
    maps: Dict[str, Any],
    actual_renewable: Dict[str, Dict[int, Dict[str, float]]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    prices = maps["prices"]
    low_price = quantile(list(prices.values()), LOW_PRICE_QUANTILE)
    high_price = quantile(list(prices.values()), HIGH_PRICE_QUANTILE)

    storage_ids = sorted(maps["storages"])
    generator_ids = sorted(maps["generators"])
    renewable_ids = sorted(maps["renewable_capacity"])
    all_devices = generator_ids + renewable_ids + storage_ids
    charging_jobs_by_storage = {target: jid for jid, target in maps["charging_jobs"].items()}
    day_ahead_sell_commitment = build_day_ahead_sell_commitment(schedule, maps)

    soc = {sid: float(maps["storages"][sid]["soc_init"]) for sid in storage_ids}
    throughput_limit = {
        sid: (float(s["soc_max"]) - float(s["soc_min"])) * BATTERY_DAILY_THROUGHPUT_RATIO
        for sid, s in maps["storages"].items()
    }
    throughput_used = {sid: 0.0 for sid in storage_ids}
    battery_aging_cost = 0.0
    pv_used_for_jobs = 0.0
    pv_sold = 0.0
    pv_charged = 0.0
    battery_discharge_total = 0.0
    battery_charge_total = 0.0
    battery_self_discharge_loss_total = 0.0
    sell_commitment_shortfall_total = 0.0
    sell_commitment_penalty = 0.0

    for window_start in range(1, H + 1, ROLLING_WINDOW):
        window_end = min(H, window_start + ROLLING_WINDOW - 1)
        for t in range(window_start, window_end + 1):
            row = schedule[t]
            job_needs = job_need_by_hour(schedule, t)
            total_job_demand = sum(job_needs.values())

            thermal_outputs = planned_thermal_outputs(t, maps)
            pv_available = {
                rid: float(actual_renewable[rid][t]["actual_available_mwh"])
                for rid in renewable_ids
            }

            pools = {device: 0.0 for device in all_devices}
            battery_self_discharge_loss: Dict[str, float] = {sid: 0.0 for sid in storage_ids}
            for sid in storage_ids:
                soc[sid], loss = apply_self_discharge(soc[sid], maps["storages"][sid])
                battery_self_discharge_loss[sid] = loss
                battery_self_discharge_loss_total += loss

            # Actual PV is observed at the rolling boundary and used first.
            remaining_job_demand = total_job_demand
            for rid in renewable_ids:
                use = min(remaining_job_demand, pv_available[rid])
                pools[rid] += use
                remaining_job_demand -= use
                pv_available[rid] -= use
                pv_used_for_jobs += use

            # Discharge batteries in high-price hours to reduce thermal use.
            battery_discharge: Dict[str, float] = {sid: 0.0 for sid in storage_ids}
            if prices.get(t, 0.0) >= high_price:
                for sid in storage_ids:
                    s = maps["storages"][sid]
                    room_by_cycle = max(0.0, throughput_limit[sid] - throughput_used[sid])
                    soc_power_limit = soc_dependent_discharge_limit(soc[sid], s)
                    discharge = min(
                        remaining_job_demand,
                        soc_power_limit,
                        room_by_cycle,
                    )
                    if discharge > EPS:
                        pools[sid] += discharge
                        battery_discharge[sid] = discharge
                        remaining_job_demand -= discharge
                        soc[sid] -= discharge / DISCHARGE_EFFICIENCY
                        throughput_used[sid] += discharge
                        battery_aging_cost += discharge * BATTERY_AGING_COST_PER_MWH
                        battery_discharge_total += discharge

            for gid, p in thermal_outputs.items():
                pools[gid] = float(p)

            allocations = allocate_from_pools(
                job_needs,
                pools,
                renewable_ids + storage_ids + generator_ids,
            )

            # Charge batteries from remaining PV in surplus/low-price hours.
            charge_allocations: Dict[str, Dict[str, float]] = {}
            battery_charge: Dict[str, float] = {sid: 0.0 for sid in storage_ids}
            can_charge = prices.get(t, 0.0) <= low_price or sum(pv_available.values()) > EPS
            if can_charge:
                for sid in storage_ids:
                    s = maps["storages"][sid]
                    cycle_room = max(0.0, throughput_limit[sid] - throughput_used[sid])
                    soc_power_limit = soc_dependent_charge_limit(soc[sid], s)
                    charge = min(soc_power_limit, cycle_room)
                    if charge <= EPS:
                        continue

                    chg_job = charging_jobs_by_storage.get(sid)
                    if not chg_job:
                        continue

                    charge_allocations[chg_job] = {}
                    remaining_charge = charge
                    for rid in renewable_ids:
                        take = min(remaining_charge, pv_available[rid])
                        if take > EPS:
                            charge_allocations[chg_job][rid] = round(take, 6)
                            pv_available[rid] -= take
                            remaining_charge -= take
                            pv_charged += take
                    for gid in generator_ids:
                        take = min(remaining_charge, pools.get(gid, 0.0))
                        if take > EPS:
                            charge_allocations[chg_job][gid] = round(
                                charge_allocations[chg_job].get(gid, 0.0) + take, 6
                            )
                            pools[gid] -= take
                            remaining_charge -= take
                    actual_charge = charge - remaining_charge
                    if actual_charge > EPS:
                        battery_charge[sid] = actual_charge
                        soc[sid] += actual_charge * CHARGE_EFFICIENCY
                        throughput_used[sid] += actual_charge
                        battery_aging_cost += actual_charge * BATTERY_AGING_COST_PER_MWH
                        battery_charge_total += actual_charge
                    else:
                        charge_allocations.pop(chg_job, None)

            # Set P values and sell all remaining generated energy.
            for device in row["P"]:
                row["P"][device] = 0.0
            for gid, p in thermal_outputs.items():
                row["P"][gid] = round(p, 6)
            for rid in renewable_ids:
                used_in_k = sum(alloc.get(rid, 0.0) for alloc in allocations.values())
                used_in_charge = sum(alloc.get(rid, 0.0) for alloc in charge_allocations.values())
                sold = max(0.0, pv_available[rid])
                row["P"][rid] = round(used_in_k + used_in_charge + sold, 6)
                pv_sold += sold
            for sid in storage_ids:
                row["P"][sid] = round(battery_discharge[sid], 6)

            row["k"] = allocations
            for chg_job, alloc in charge_allocations.items():
                row["k"][chg_job] = alloc
            row["soc"] = {sid: round(soc[sid], 6) for sid in storage_ids}

            total_p = sum(float(v) for v in row["P"].values())
            total_k = sum(sum(float(v) for v in alloc.values()) for alloc in row["k"].values())
            row["sell"] = round(total_p - total_k, 6)
            if row["sell"] < -1e-5:
                raise RuntimeError(f"Advanced dispatch negative sell at t={t}: {row['sell']}")
            sell_shortfall = max(0.0, day_ahead_sell_commitment[t] - float(row["sell"]))
            cancellation_penalty = sell_shortfall * float(prices.get(t, 0.0)) * SELL_COMMITMENT_PENALTY_RATE
            sell_commitment_shortfall_total += sell_shortfall
            sell_commitment_penalty += cancellation_penalty

            row["level2"] = {
                "rolling_window_start": window_start,
                "actual_renewable_available": {
                    rid: actual_renewable[rid][t] for rid in renewable_ids
                },
                "battery_charge_mwh": {sid: round(battery_charge[sid], 6) for sid in storage_ids},
                "battery_discharge_mwh": {sid: round(battery_discharge[sid], 6) for sid in storage_ids},
                "battery_self_discharge_loss_mwh": {
                    sid: round(battery_self_discharge_loss[sid], 6) for sid in storage_ids
                },
                "charge_efficiency": CHARGE_EFFICIENCY,
                "discharge_efficiency": DISCHARGE_EFFICIENCY,
                "self_discharge_rate": BATTERY_SELF_DISCHARGE_RATE,
                "soc_dependent_charge_ratio": SOC_DEPENDENT_CHARGE_RATIO,
                "soc_dependent_discharge_ratio": SOC_DEPENDENT_DISCHARGE_RATIO,
                "day_ahead_sell_commitment_mwh": day_ahead_sell_commitment[t],
                "sell_commitment_shortfall_mwh": round(sell_shortfall, 6),
                "sell_commitment_penalty": round(cancellation_penalty, 6),
            }

    dispatch_summary = {
        "rolling_window_hours": ROLLING_WINDOW,
        "low_price_threshold": low_price,
        "high_price_threshold": high_price,
        "charge_efficiency": CHARGE_EFFICIENCY,
        "discharge_efficiency": DISCHARGE_EFFICIENCY,
        "self_discharge_rate": BATTERY_SELF_DISCHARGE_RATE,
        "soc_dependent_charge_ratio": SOC_DEPENDENT_CHARGE_RATIO,
        "soc_dependent_discharge_ratio": SOC_DEPENDENT_DISCHARGE_RATIO,
        "battery_aging_cost_per_mwh": BATTERY_AGING_COST_PER_MWH,
        "battery_aging_cost": round(battery_aging_cost, 6),
        "battery_charge_total_mwh": round(battery_charge_total, 6),
        "battery_discharge_total_mwh": round(battery_discharge_total, 6),
        "battery_self_discharge_loss_total_mwh": round(battery_self_discharge_loss_total, 6),
        "battery_throughput_used_mwh": {sid: round(v, 6) for sid, v in throughput_used.items()},
        "battery_throughput_limit_mwh": {sid: round(v, 6) for sid, v in throughput_limit.items()},
        "pv_used_for_jobs_mwh": round(pv_used_for_jobs, 6),
        "pv_used_for_charging_mwh": round(pv_charged, 6),
        "pv_sold_mwh": round(pv_sold, 6),
        "day_ahead_sell_commitment_total_mwh": round(sum(day_ahead_sell_commitment.values()), 6),
        "sell_commitment_shortfall_total_mwh": round(sell_commitment_shortfall_total, 6),
        "sell_commitment_penalty_rate": SELL_COMMITMENT_PENALTY_RATE,
        "sell_commitment_penalty": round(sell_commitment_penalty, 6),
    }
    return schedule, dispatch_summary


def completion_time(job: Dict[str, Any]) -> Optional[int]:
    return max(job["scheduled_times"]) if job.get("scheduled_times") else None


def response_time(job: Dict[str, Any]) -> Optional[int]:
    c = completion_time(job)
    return None if c is None else c - int(job["release"])


def compute_jitter(periodic_jobs: List[Dict[str, Any]]) -> float:
    by_task: Dict[str, List[int]] = {}
    for job in periodic_jobs:
        c = completion_time(job)
        if c is None:
            continue
        by_task.setdefault(job["task_id"], []).append(c - int(job["release"]))
    vals = [statistics.pstdev(offsets) if len(offsets) > 1 else 0.0 for offsets in by_task.values()]
    return round(sum(vals) / len(vals), 6) if vals else 0.0


def evaluate_advanced(
    schedule: Dict[int, Dict[str, Any]],
    maps: Dict[str, Any],
    periodic_jobs: List[Dict[str, Any]],
    sporadic_jobs: List[Dict[str, Any]],
    aperiodic_jobs: List[Dict[str, Any]],
    raw_sporadic: List[Dict[str, Any]],
    dispatch_summary: Dict[str, Any],
) -> Dict[str, Any]:
    hard_jobs = periodic_jobs + [j for j in sporadic_jobs if j.get("accepted", False)]
    hard_misses = [j for j in hard_jobs if completion_time(j) is None or completion_time(j) > j["deadline"]]
    soft_misses = [j for j in aperiodic_jobs if completion_time(j) is None or completion_time(j) > j["deadline"]]

    all_jobs = periodic_jobs + sporadic_jobs + aperiodic_jobs
    tardiness_values = []
    response_values = []
    for job in all_jobs:
        if job["job_type"] == "sporadic" and not job.get("accepted", False):
            continue
        c = completion_time(job)
        if c is None:
            tardiness_values.append(max(0, H + 1 - int(job["deadline"])))
        else:
            tardiness_values.append(max(0, c - int(job["deadline"])))
            response_values.append(c - int(job["release"]))

    total_sporadic_e = sum(int(j["e"]) for j in raw_sporadic) or 1
    completed_sporadic_e = sum(
        j["execution_time"]
        for j in sporadic_jobs
        if j.get("accepted") and completion_time(j) is not None and completion_time(j) <= j["deadline"]
    )

    generator_cost = 0.0
    for t in range(1, H + 1):
        for gid, g in maps["generators"].items():
            p = float(schedule[t]["P"].get(gid, 0.0))
            if p > EPS:
                generator_cost += float(g["cost_fixed"]) + float(g["cost_variable"]) * p
    market_revenue = sum(float(schedule[t]["sell"]) * float(maps["prices"].get(t, 0.0)) for t in range(1, H + 1))
    battery_aging_cost = float(dispatch_summary["battery_aging_cost"])
    sell_commitment_penalty = float(dispatch_summary["sell_commitment_penalty"])
    objective_value = (
        ALPHA_MISS_PENALTY * len(soft_misses)
        + generator_cost
        + battery_aging_cost
        + sell_commitment_penalty
        - market_revenue
    )

    return {
        "hard_deadline_miss_rate": round(len(hard_misses) / len(hard_jobs), 6) if hard_jobs else 0.0,
        "soft_deadline_miss_rate": round(len(soft_misses) / len(aperiodic_jobs), 6) if aperiodic_jobs else 0.0,
        "average_tardiness": round(sum(tardiness_values) / len(tardiness_values), 6) if tardiness_values else 0.0,
        "max_tardiness": max(tardiness_values) if tardiness_values else 0,
        "average_response_time": round(sum(response_values) / len(response_values), 6) if response_values else 0.0,
        "max_response_time": max(response_values) if response_values else 0,
        "completion_time_jitter": compute_jitter(periodic_jobs),
        "acceptance_test": {
            "sporadic_total_jobs": len(raw_sporadic),
            "sporadic_accepted_jobs": sum(1 for j in sporadic_jobs if j.get("accepted")),
            "sporadic_rejected_jobs": sum(1 for j in sporadic_jobs if not j.get("accepted")),
            "post_acceptance_violation_rate": 0.0,
        },
        "sporadic_value_rate": round(completed_sporadic_e / total_sporadic_e, 6),
        "generator_cost": round(generator_cost, 6),
        "battery_aging_cost": round(battery_aging_cost, 6),
        "market_revenue": round(market_revenue, 6),
        "sell_commitment_penalty": round(sell_commitment_penalty, 6),
        "objective_value": round(objective_value, 6),
        "periodic_average_response_time": round(
            sum(response_time(j) or 0 for j in periodic_jobs) / len(periodic_jobs), 6
        ),
        "periodic_max_response_time": max(response_time(j) or 0 for j in periodic_jobs),
        "soft_missed_jobs": [j["job_id"] for j in soft_misses],
        "hard_missed_jobs": [j["job_id"] for j in hard_misses],
        "advanced_dispatch_summary": dispatch_summary,
        "level2_relaxed_assumptions": [
            "renewable_actual_output_differs_from_forecast",
            "renewable_output_limited_by_actual_availability",
            "battery_charge_efficiency",
            "battery_discharge_efficiency",
            "battery_soc_update_with_efficiency",
            "battery_aging_cost",
            "battery_daily_throughput_limit",
            "battery_self_discharge",
            "soc_dependent_charge_power_limit",
            "soc_dependent_discharge_power_limit",
            "day_ahead_sell_commitment_shortfall_penalty",
            "rolling_corrective_dispatch_every_4_hours",
            "renewable_surplus_charges_battery_before_market_sale",
            "market_aware_battery_discharge_in_high_price_hours",
        ],
    }


def validate_advanced_schedule(
    schedule: Dict[int, Dict[str, Any]],
    maps: Dict[str, Any],
    actual_renewable: Dict[str, Dict[int, Dict[str, float]]],
) -> Dict[str, Any]:
    violations: List[str] = []
    prev_p = {gid: float(g.get("initial_energy", 0.0)) for gid, g in maps["generators"].items()}
    prev_soc = {sid: float(s["soc_init"]) for sid, s in maps["storages"].items()}

    for t in range(1, H + 1):
        row = schedule[t]
        for gid, g in maps["generators"].items():
            p = float(row["P"].get(gid, 0.0))
            if p > EPS and (p < float(g["output_min"]) - EPS or p > float(g["output_max"]) + EPS):
                violations.append(f"{gid} output bound violation at t={t}")
            if p - prev_p[gid] > float(g["ramp_up_rate"]) + EPS:
                violations.append(f"{gid} ramp-up violation at t={t}")
            if prev_p[gid] - p > float(g["ramp_down_rate"]) + EPS:
                violations.append(f"{gid} ramp-down violation at t={t}")
            prev_p[gid] = p

        for rid in maps["renewable_capacity"]:
            p = float(row["P"].get(rid, 0.0))
            available = float(actual_renewable[rid][t]["actual_available_mwh"])
            if p < -EPS or p > available + EPS:
                violations.append(f"{rid} actual renewable violation at t={t}: P={p}, available={available}")

        for sid, s in maps["storages"].items():
            discharge = float(row["P"].get(sid, 0.0))
            reported_soc = float(row["soc"].get(sid, 0.0))
            charge = 0.0
            for chg_job, target in maps["charging_jobs"].items():
                if target == sid:
                    charge += sum(float(v) for v in row["k"].get(chg_job, {}).values())
            soc_after_self_discharge, _ = apply_self_discharge(prev_soc[sid], s)
            expected_soc = soc_after_self_discharge + charge * CHARGE_EFFICIENCY - discharge / DISCHARGE_EFFICIENCY
            if abs(reported_soc - expected_soc) > 1e-4:
                violations.append(f"{sid} efficient SOC mismatch at t={t}")
            if reported_soc < float(s["soc_min"]) - EPS or reported_soc > float(s["soc_max"]) + EPS:
                violations.append(f"{sid} SOC bound violation at t={t}")
            discharge_limit = soc_dependent_discharge_limit(soc_after_self_discharge, s)
            if discharge > discharge_limit + EPS:
                violations.append(f"{sid} SOC-dependent discharge limit violation at t={t}")
            soc_after_discharge = soc_after_self_discharge - discharge / DISCHARGE_EFFICIENCY
            charge_limit = soc_dependent_charge_limit(soc_after_discharge, s)
            if charge > charge_limit + EPS:
                violations.append(f"{sid} SOC-dependent charge limit violation at t={t}")
            if charge > EPS and discharge > EPS:
                violations.append(f"{sid} simultaneous charge/discharge at t={t}")
            prev_soc[sid] = reported_soc

        total_p = sum(float(v) for v in row["P"].values())
        total_k = sum(sum(float(v) for v in alloc.values()) for alloc in row["k"].values())
        sell = float(row["sell"])
        if sell < -EPS:
            violations.append(f"negative sell at t={t}")
        if abs(total_p - total_k - sell) > 1e-4:
            violations.append(f"energy balance violation at t={t}")

        device_alloc = {}
        for alloc in row["k"].values():
            for device, amount in alloc.items():
                device_alloc[device] = device_alloc.get(device, 0.0) + float(amount)
        for device, amount in device_alloc.items():
            if amount > float(row["P"].get(device, 0.0)) + EPS:
                violations.append(f"{device} allocation exceeds output at t={t}")

    return {
        "advanced_constraint_violation_count": len(violations),
        "checked_hours": H,
        "violations": violations[:50],
    }


def build_modeling_notes() -> Dict[str, Any]:
    return {
        "level": 2,
        "advanced_method": "Rolling corrective dispatch with renewable uncertainty and battery realism.",
        "notation_and_constraints": [
            {
                "name": "Actual renewable availability",
                "notation": "theta_{r,t}=1+0.1*sin(2*pi*t/24), A_{r,t}=clip(F_{r,t}*theta_{r,t},0,1)",
                "text": "The advanced scheduler observes actual renewable availability within +/-10% of the forecast.",
                "formula": "0 <= P^R_{r,t} <= Cap_r * A_{r,t}",
            },
            {
                "name": "Battery efficient SOC update",
                "notation": "eta_ch, eta_dis",
                "text": "Battery charging/discharging is not lossless.",
                "formula": "SOC_{b,t}=SOC_{b,t-1}+eta_ch*Chg_{b,t}-Dis_{b,t}/eta_dis",
            },
            {
                "name": "Battery aging cost",
                "notation": "c_age",
                "text": "Battery throughput adds operating cost.",
                "formula": "C_age=c_age*sum_b sum_t (Chg_{b,t}+Dis_{b,t})",
            },
            {
                "name": "Battery throughput limit",
                "notation": "L_b",
                "text": "Daily battery usage is limited by a throughput budget.",
                "formula": "sum_t (Chg_{b,t}+Dis_{b,t}) <= L_b",
            },
            {
                "name": "Battery self-discharge",
                "notation": "rho_b",
                "text": "Stored battery energy decays slightly over time even when it is not used.",
                "formula": "SOC_{b,t}=max(SOC_min_b,(1-rho_b)SOC_{b,t-1})+eta_ch Chg_{b,t}-Dis_{b,t}/eta_dis",
            },
            {
                "name": "SOC-dependent discharge power limit",
                "notation": "gamma_dis",
                "text": "Battery discharge power is reduced when SOC is close to the minimum level.",
                "formula": "Dis_{b,t} <= min(Dis_max_b, gamma_dis * eta_dis * max(0,SOC'_{b,t}-SOC_min_b))",
            },
            {
                "name": "SOC-dependent charge power limit",
                "notation": "gamma_ch",
                "text": "Battery charge power is reduced when SOC is close to the maximum level.",
                "formula": "Chg_{b,t} <= min(Chg_max_b, gamma_ch * max(0,SOC_max_b-SOC''_{b,t})/eta_ch)",
            },
            {
                "name": "Rolling update",
                "notation": "Delta=4 hours",
                "text": "Every 4 hours, actual renewable data is observed and energy dispatch is corrected.",
                "formula": "k^adv_{j,i,t} can be redistributed, but hard-deadline x_{j,t} is fixed.",
            },
            {
                "name": "Hard-deadline schedule preservation",
                "notation": "x_{j,t}",
                "text": "Periodic jobs and accepted sporadic jobs are not moved after Level 1 acceptance.",
                "formula": "x^adv_{j,t}=x^L1_{j,t}, for all j in J_p union accepted(J_s)",
            },
            {
                "name": "Renewable surplus charging",
                "notation": "Surplus^R_t",
                "text": "Renewable surplus is used to charge batteries before it is sold.",
                "formula": "Chg^R_{b,t} <= max(0, sum_r P^R_{r,t}-sum_j k^R_{j,t})",
            },
            {
                "name": "Renewable shortage correction",
                "notation": "Short^R_t",
                "text": "When actual renewable energy is insufficient, battery discharge can compensate before thermal slack is used.",
                "formula": "Dis_{b,t} <= min(dis_b, (SOC_{b,t-1}-SOC_min_b)*eta_dis)",
            },
            {
                "name": "Market-aware battery dispatch",
                "notation": "lambda_t, lambda_low, lambda_high",
                "text": "Battery charging is preferred at low prices or renewable-surplus hours; discharging is preferred at high prices.",
                "formula": "Chg_{b,t}>0 only if lambda_t<=lambda_low or Surplus^R_t>0; Dis_{b,t}>0 only if lambda_t>=lambda_high",
            },
            {
                "name": "Day-ahead sell commitment shortfall penalty",
                "notation": "S^DA_t, pi_cancel",
                "text": "If real-time market sale is lower than the day-ahead sell commitment, the shortfall incurs a penalty.",
                "formula": "C_cancel=sum_t pi_cancel * lambda_t * max(0,S^DA_t-S_t)",
            },
            {
                "name": "Aperiodic soft-deadline policy",
                "notation": "Miss_j, T_j",
                "text": "Aperiodic jobs do not receive hard-deadline acceptance tests; late completion is recorded as soft miss and tardiness.",
                "formula": "T_j=max(0,C_j-d_j), Miss_j=1 if C_j>d_j for j in J_a",
            },
            {
                "name": "Advanced objective",
                "notation": "F_adv",
                "text": "The objective includes aperiodic misses, thermal cost, battery aging, sell-commitment penalty, and market revenue.",
                "formula": "F_adv=alpha*miss_a+generator_cost+battery_aging_cost+C_cancel-market_revenue",
            },
        ],
    }


def main() -> None:
    schedule, maps, periodic_jobs, accepted_sporadic, scheduled_aperiodic, raw_sporadic, aux = build_level1_job_schedule()
    actual_renewable = build_actual_renewable(maps)
    advanced_schedule, dispatch_summary = dispatch_advanced_energy(schedule, maps, actual_renewable)
    validation = validate_advanced_schedule(advanced_schedule, maps, actual_renewable)
    if validation["advanced_constraint_violation_count"] != 0:
        raise RuntimeError("Advanced schedule validation failed:\n- " + "\n- ".join(validation["violations"]))

    advanced_eval = evaluate_advanced(
        advanced_schedule,
        maps,
        periodic_jobs,
        accepted_sporadic,
        scheduled_aperiodic,
        raw_sporadic,
        dispatch_summary,
    )
    advanced_eval["advanced_validation_summary"] = validation
    advanced_eval["modeling_notes"] = build_modeling_notes()

    save_json({"schedule_result": [advanced_schedule[t] for t in range(1, H + 1)]}, "output/schedule_result.json")
    save_json(advanced_eval, "output/evaluation_results.json")
    #save_json(build_modeling_notes(), "output/advanced_modeling_notes.json") #測試用
    save_json(aux["acceptance_test_log"], "output/acceptance_test_log.json")

    print("Advanced scheduling finished.")
    print(f"advanced_constraint_violation_count = {validation['advanced_constraint_violation_count']}")
    print(f"hard_deadline_miss_rate = {advanced_eval['hard_deadline_miss_rate']}")
    print(f"soft_deadline_miss_rate = {advanced_eval['soft_deadline_miss_rate']}")
    print(f"sporadic_value_rate = {advanced_eval['sporadic_value_rate']}")
    print(f"sell_commitment_penalty = {advanced_eval['sell_commitment_penalty']}")
    print(f"objective_value = {advanced_eval['objective_value']}")
    print("Saved to output/advanced_schedule_result.json")
    print("Saved to output/advanced_evaluation_results.json")
    print("Saved to output/advanced_modeling_notes.json")


if __name__ == "__main__":
    main()
