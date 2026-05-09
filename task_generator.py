import random
import json
import math

# 這個只是用來驗證生成的任務集是否符合規定的函式，實際上在生成任務集的過程中已經盡量確保符合規定了
def final_validate_tasks(tasks):
    # 1-1: 要有jobID, release time, period, executime time, energy demand, relative deadline, preemptive/non-preemptive
    for task_id, t in tasks.items():
        if not all(k in t for k in ['r', 'p', 'e', 'd', 'w', 'preempt']):
            print("Missing required fields in task:", task_id)
            return False

    # 1-2: 6 <= num_tasks <= 10
    num_tasks = len(tasks)
    if not (6 <= num_tasks <= 10):
        print("Number of tasks must be between 6 and 10. Found:", num_tasks)
        return False
    
    # 1-3: 展開後的 periodic jobs 數量必須大於 30 個。
    total_jobs = sum(72 // t['p'] for t in tasks.values())
    if total_jobs <= 30:
        print("Total number of periodic jobs must be greater than 30. Found:", total_jobs)
        return False
    
    # 1-4: 1 <= r_j <= p
    for task_id, t in tasks.items():
        if not (1 <= t['r'] <= t['p']):
            print(f"Task {task_id}: Release time r must be between 1 and period p. Found r={t['r']}, p={t['p']}")
            return False
    
    # 1-4: 6 <= p_j <= 24
    for task_id, t in tasks.items():
        if not (6 <= t['p'] <= 24):
            print(f"Task {task_id}: Period p must be between 6 and 24. Found p={t['p']}")
            return False
    
    # 1-4: 1 <= e_j <= 4，至少2個periodic task的e_j = 2，至少1個periodic tasks的e_j >= 3
    e_values = [t['e'] for t in tasks.values()]
    if not all(1 <= e <= 4 for e in e_values):
        print("Execution time e must be between 1 and 4 for all tasks. Found:", e_values)
        return False
    if e_values.count(2) < 2:
        print("At least 2 periodic tasks must have execution time e = 2. Found count:", e_values.count(2))
        return False
    if sum(1 for e in e_values if e >= 3) < 1:
        print("At least 1 periodic task must have execution time e >= 3. Found count:", sum(1 for e in e_values if e >= 3))
        return False
    
    # 1-5: Periodic workload density 規定
    dw = sum(t['e'] / t['p'] for t in tasks.values())
    if not (0.7 <= dw <= 1.0):
        print(f"Periodic workload density dw must be between 0.7 and 1.0. Found dw={dw:.4f}")
        return False
    
    # 1-6: 至少20%的periodic tasks符合 d = e
    d_equals_e_count = sum(1 for t in tasks.values() if t['d'] == t['e'])
    if d_equals_e_count < math.ceil(num_tasks * 0.2):
        print(f"At least 20% of periodic tasks must satisfy d = e. Found count: {d_equals_e_count}, required: {math.ceil(num_tasks * 0.2)}")
        return False
    
    # 1-7: Non-preemptive tasks: 至少2個e != 1的periodic tasks為non-preemptive
    np_candidates = [t for t in tasks.values() if t['e'] > 1]
    np_count = sum(1 for t in np_candidates if t['preempt'] == 0)
    if np_count < 2:
        print(f"At least 2 non-preemptive tasks with e != 1 are required. Found count: {np_count}")
        return False
    
    # 1-8: Frame size 規定:
    # 若裡面沒有放selected_f，則不檢查F的規定
    is_selected_f_in_tasks = all('selected_f' in t for t in tasks.values())
    if is_selected_f_in_tasks:
        max_e = max(t['e'] for t in tasks.values())
        min_p = min(t['p'] for t in tasks.values())
        f = tasks[next(iter(tasks))]['selected_f']
        H = 72
        
        if H % f != 0:
            print(f"Selected frame size F must be a factor of 72. Found F={f} which does not divide 72.")
            return False

        if f < max_e or f > min_p:
            print(f"Selected frame size F must be >= max(e) and <= min(p). Found F={f}, max(e)={max_e}, min(p)={min_p}")
            return False
        
        for task_id, t in tasks.items():
            if 2 * f - math.gcd(f, t['p']) > t['d']:
                print(f"Task {task_id}: Frame size F does not satisfy the condition 2F - gcd(F, p) <= d. Found F={f}, p={t['p']}, d={t['d']}")
                return False
    return True

def validate_tasks(tasks):
    # 1-3: 展開後的 periodic jobs 數量必須大於 30 個。
    total_jobs = sum(72 // t['p'] for t in tasks.values())
    if total_jobs <= 30:
        return False
    
    # 1-4: 至少3種period值
    periods = set(t['p'] for t in tasks.values())
    if len(periods) < 3:
        return False
    
    # 1-5: Periodic workload density 規定
    dw = sum(t['e'] / t['p'] for t in tasks.values())
    if not (0.7 <= dw <= 1.0):
        return False
    
    # 1-8: Frame size 規定:
    # (F must be a factor of 72, F >= max(e), and F <= min(p))
    valid_f = None
    max_e = max(t['e'] for t in tasks.values())
    min_p = min(t['p'] for t in tasks.values())
    for f in [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72]:
        if f < max_e or f > min_p:
            continue

        is_f_ok = True
        for t in tasks.values():
            if 2 * f - math.gcd(f, t['p']) > t['d']:
                is_f_ok = False
                break
        
        if is_f_ok:
            valid_f = f
            break
    
    if valid_f is None:
        return False
    
    for t in tasks.values():
        t['selected_f'] = valid_f

    return True

def _try_to_generate_period_tasks():
    tasks = {}

    num_tasks  = random.randint(6, 10)
    for i in range(1, num_tasks + 1):
        task_id = f"p{i}"
        # 1-4. 6 <= p_j <= 24
        # 為讓gcd變大，優先選擇72的因數作為period值
        p = random.randint(6, 24)

        # 1-4. 1 <= e_j <= 4
        # 至少2個periodic task的e_j = 2
        # 至少1個periodic tasks的e_j >= 3
        if i < 3:
            e = 2
        elif i == 3:
            e = 3
        else:
            e = random.randint(1, 4)

        d = random.randint(e, p)
        r = random.randint(1, p)
        w = random.randint(6, 18)
        preempt = random.choice([0, 1])
        
        tasks[task_id] = {
            "r": r, "p": p, "e": e, "d": d, "w": w, "preempt": preempt
        }

    task_list  = list(tasks.values())
    indices = list(range(len(task_list)))
    random.shuffle(indices)

    # 1-4: 至少2個periodic tasks的w >= 14
    w_vals = [t['w'] for t in task_list]
    high_w_count = sum(1 for w in w_vals if w >= 14)
    if high_w_count < 2:
        for i in range(0, 2 - high_w_count):
            task_list[indices[i]]['w'] = random.randint(14, 18)

    # 1-6: 至少20%的periodic tasks符合 d = e
    for i in range(0, math.ceil(num_tasks * 0.2)):
        task_list[indices[i]]['d'] = task_list[indices[i]]['e']

    # 1-7: Non-preemptive tasks: 至少2個e != 1的periodic tasks為non-preemptive
    np_candidates = [idx for idx in indices if task_list[idx]['e'] > 1]
    random.shuffle(np_candidates)
    task_list[np_candidates[0]]['preempt'] = 0
    task_list[np_candidates[1]]['preempt'] = 0

    final_tasks_dict = {f"p{i+1}": task_list[i] for i in range(num_tasks)}
    return final_tasks_dict

def generate_periodic_tasks():
    while True:
        tasks = _try_to_generate_period_tasks()
        if validate_tasks(tasks):
            return tasks

if __name__ == "__main__":
    periodic_tasks = generate_periodic_tasks()

    # for task in periodic_tasks.values():
    #     if 'selected_f' in task:
    #         del task['selected_f']

    print(json.dumps({"periodic": periodic_tasks}, indent=2))
    # Save Files
    with open('../output/task_set.json', 'w') as f:
        json.dump({"periodic": periodic_tasks}, f, indent=2)
        
    print("task_set.json has created successfully in output directory.")
    print(final_validate_tasks(periodic_tasks))