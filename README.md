# README.md

本專案實作了一套虛擬電廠排程系統，包含日前固定排程（Level 1）與考量再生能源波動及非理想電池模型的進階動態滾動排程（Level 2）。

## 使用語言、版本與套件需求
* **程式語言**: Python
* **版本要求**: Python 3.10 或以上版本
* **套件要求**: 僅使用原生標準庫 `json`, `pathlib`, `typing`, `math`, `statistics`（無須安裝任何第三方依賴套件）。
* **作業系統**: Windows / macOS / Linux

## 目錄架構要求
請確保專案維持以下標準架構，程式方能正確讀取輸入與寫入輸出：
```text
├── input/
│   ├── processor_settings.json
│   ├── price_72hr.json
│   └── aperiodic_n_sporadic.json
├── output/
│   ├── task_set.json
│   ├── schedule_result.json
│   ├── acceptance_test_log.json
│   └── evaluation_results.json
└── src/
    ├── task_generator.py
    ├── scheduler.py
    ├── evaluator.py
    └── advanced_scheduler.py

```

---
## 程式間依賴關係

各程式之資料流如下：

level 1：
```text
                         input/
        ┌─────────────────────────────────────┐
        │ processor_settings.json             │
        │ price_72hr.json                     │
        │ aperiodic_n_sporadic.json           │
        └───────────────┬─────────────────────┘
                        │                     
                        │                     
    task_generator.py   │                     
        │               │                     
        ▼               │                     
     task_set.json      │                     
        │               │                     
        └──────┬───────┬┘                     
               ▼       └──────────────────────┐                     
                                              │
         scheduler.py                         │
               │                              │
               ├── schedule_result.json       │
               └── acceptance_test_log.json   │
                          │                   │
                          └──────┬────────────┘
                                 ▼
                            evaluator.py
                             │
                             ▼

             evaluation_results.json

```


Level 2 流程：

```text
                         input/
        ┌─────────────────────────────────────┐
        │ processor_settings.json             │
        │ price_72hr.json                     │
        │ aperiodic_n_sporadic.json           │
        └───────────────┬─────────────────────┘
                        │
                        │
task_generator.py       │
        │               │
        ▼               │
 task_set.json          │
        │               │
        └──────┬────────┘
               ▼

      advanced_scheduler.py
               │
               ├── schedule_result.json
               ├── acceptance_test_log.json
               └── evaluation_results.json
```

---


## 程式執行流程 (操作指南)

在終端機中，將當前路徑切換至**專案根目錄**（即包含 `src/` 與 `input/` 的資料夾），並依序執行以下指令以重現所有結果：

---
**level 1:**

### Step 1: 隨機生成合法週期性任務

```bash
python src/task_generator.py
```

* **功能**：自動產生符合嚴格數學限制式的週期任務。
* **產出**：`output/task_set.json`

### Step 2: 執行 Level 1 基礎排程 (日前排程)

```bash
python src/scheduler.py
```

* **功能**：讀取任務與設備參數，進行日前排程。
* **產出**：
`output/schedule_result.json`、`output/acceptance_test_log.json`

### Step 3: 執行 Level 1 獨立評估驗證

```bash
python src/evaluator.py
```

* **功能**：以第三方評估器視角，審查排程結果是否違反 23 項物理與時間限制式，並結算目標函數（Objective Value）。
* **產出**：`output/evaluation_results.json`


---
**level 2**

### Step 1: 隨機生成合法週期性任務

```bash
python src/task_generator.py
```

* **功能**：自動產生符合嚴格數學限制式的週期任務。
* **產出**：`output/task_set.json`

### Step 2: 執行 Level 2 進階動態滾動排程與驗證

```bash
python src/advanced_scheduler.py
```

* **功能**：載入再生能源 $\pm10\%$ 不確定性與非理想電池模型，每 3 小時進行一次 Rolling Corrective Dispatch，並**自動完成 evaluation** 。
* **產出**：
`output/schedule_result.json`、`output/acceptance_test_log.json`、`output/evaluation_results.json`

* [註] advanced_scheduler 輸出會直接覆蓋level1的版本

---

## 如何重現繳交的 Output JSON

本次繳交之輸出檔案可透過以下流程重新產生。

### level 1

由於任務產生器採用固定亂數種子：

```
SEED = 42
```

因此在不修改程式與輸入檔案的情況下，每次執行皆會產生相同的：

```
output/task_set.json
```

重現流程如下：

```
python src/task_generator.py
python src/scheduler.py
python src/evaluator.py
```

即可重新產生：

```
output/task_set.json
output/schedule_result.json
output/acceptance_test_log.json
output/evaluation_results.json
```

其內容應與繳交版本一致。

---

###  Level 2 


執行：

```
python src/task_generator.py
python src/advanced_scheduler.py
```

即可重新產生：

```
output/task_set.json
output/schedule_result.json
output/acceptance_test_log.json
output/evaluation_results.json
```

其中：

```
advanced_scheduler.py
```

會自動完成排程與評估，不需額外執行 evaluator.py。

---

## 各程式輸入與輸出檔案說明


### 1. 輸入檔案說明 (`input/`)

* **`processor_settings.json`**：電力系統物理設備參數設定。包含傳統火力機組（上下限、升降率、成本）、再生能源容量、預測發電百分比，以及儲能電池參數（SOC 限制、最大充放電率）。
* **`price_72hr.json`**：電力市場日前預測電價。包含 1 到 72 小時的電力市場收購價格，供系統評估套利。
* **`aperiodic_n_sporadic.json`**：非週期任務清單。




### 2. 輸出檔案說明 (`output/`)

* **`task_set.json`**：`task_generator.py` 生成的週期性任務集合，內含 $r$ (釋放時間), $p$ (週期), $e$ (執行時間), $d$ (時限), $w$ (耗能), $preempt$ (搶佔屬性)。
* **`schedule_result.json`**：排程器產出的 72 小時 電力與任務調度總表。內含每小時發電端出力 ($P$)、用電端分配 ($k$)、電池狀態 ($soc$) 與淨售電量 ($sell$)。
* **`acceptance_test_log.json`**：Sporadic task 的acceptance test決策日誌。
* **`evaluation_results.json`**：全面性驗證報告。包含時限錯失率(hard/soft deadline miss rate)、響應時間(response time)、動態准入效能(sporadic value rate 等等)、財務指標（發電成本與營收）、目標函數值與限制式違規摘要。


---
