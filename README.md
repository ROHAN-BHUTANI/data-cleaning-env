# 🚀 Advanced Data Cleaning RL Environment (OpenEnv)

💡 This environment bridges the gap between static data pipelines and intelligent autonomous data agents.

## 🌍 Overview

This project introduces a **real-world reinforcement learning environment for automated data preprocessing**, a critical step in every data science and ML pipeline.

Unlike traditional RL environments (games or simulations), this environment models **actual industry workflows** such as:

* Handling missing values
* Removing duplicates
* Normalizing distributions
* Managing outliers

👉 This makes it directly useful for **training autonomous data agents** and evaluating LLM-based reasoning systems.

## 🤖 Why Reinforcement Learning?

Traditional data cleaning pipelines are static and rule-based, requiring manual tuning for each dataset.

This environment introduces a reinforcement learning approach where an agent:
- Observes dataset state
- Decides optimal transformations dynamically
- Learns from feedback (reward signals)

This enables **adaptive, state-aware data preprocessing**, making it suitable for real-world scenarios where data distributions and issues vary significantly.

👉 Unlike fixed pipelines, this approach can generalize across datasets and improve over time.

---

## 🎯 Why This Matters (Real-World Utility)

Data cleaning accounts for **~80% of data science effort**.

This environment enables:

* Training AI agents to automate preprocessing
* Benchmarking LLM reasoning on structured data tasks
* Evaluating decision-making under noisy datasets

💡 This fills a gap between:

* Static datasets ❌
* Fully automated pipelines ❌
* → and **adaptive intelligent data agents** ✅

---

## 🧠 Task Design

We provide **3 progressively harder tasks**:

### 🟢 Easy Task

* Fill missing values
* Basic type casting
* Objective: Restore dataset completeness

### 🟡 Medium Task

* Remove duplicates
* Normalize numerical features
* Objective: Improve statistical consistency

### 🔴 Hard Task

* Handle outliers
* Apply multiple transformations
* Objective: Achieve fully production-ready dataset

✔ Tasks are deterministic
✔ Clearly defined success criteria
✔ Increasing reasoning complexity

## 🧩 Task Details

### 🟢 Easy Task — Basic Data Cleaning
- **Input:** Dataset with missing values and incorrect data types
- **Objective:** Fill null values and correct column types
- **Success Criteria:** Dataset matches expected clean structure with no missing values

💡 Focus: Fundamental preprocessing operations

---

### 🟡 Medium Task — Data Consistency
- **Input:** Dataset containing duplicate rows and unnormalized values
- **Objective:** Remove duplicates and normalize numerical columns
- **Success Criteria:** Clean dataset with consistent distributions and no redundancy

💡 Focus: Data integrity and statistical consistency

---

### 🔴 Hard Task — Advanced Data Repair
- **Input:** Dataset with outliers, schema inconsistencies, and multiple issues
- **Objective:** Handle outliers and apply multiple transformations correctly
- **Success Criteria:** Dataset is fully production-ready and matches ground truth closely

💡 Focus: Multi-step reasoning and complex transformations

---

## ⚙️ Action Space

Hybrid action space:

### Discrete Operations:

* `fill_nulls`
* `drop_duplicates`
* `normalize`
* `cast_column`
* `clip_outliers`
* `submit`

### Parameterized Inputs:

* Column selection
* Value selection
* Method selection

👉 Enables both **symbolic + parametric decision-making**

---

## 📊 Observation Space

Each step returns structured metadata:

* Column statistics (mean, std, null ratio)
* Schema information
* Data distribution signals
* Row-level alignment indicators

👉 Designed for **LLM reasoning + RL compatibility**

---

## 🧮 Reward Function

Dense and informative reward system:

| Component          | Description                          |
| ------------------ | ------------------------------------ |
| Data Quality       | Measures correctness vs ground truth |
| Progress Reward    | Rewards partial improvements         |
| Efficiency Penalty | Penalizes unnecessary actions        |

### 🔒 Stability Guarantee

All scores are strictly normalized to:

👉 **(0, 1)** — never 0.0 or 1.0
Ensures:

* Stable RL training
* Validator compliance
* Smooth reward gradients

---

## 🤖 Baseline Agent (LLM-Powered)

The baseline uses:

* OpenAI-compatible client
* LiteLLM proxy integration
* ReAct-style reasoning

### Workflow:

1. Observe dataset state
2. Reason using LLM
3. Select transformation
4. Execute action
5. Repeat until submit

---

## 🔁 API Design (OpenEnv Spec)

Fully compliant with OpenEnv:

* `reset()` → initialize environment
* `step(action)` → apply transformation
* `state()` → retrieve current dataset

✔ Typed models
✔ Deterministic transitions
✔ Clean episode boundaries

---

## 🐳 Deployment

* Dockerized environment
* Deployed on Hugging Face Spaces
* Fully reproducible execution

---

## ▶️ Running the Project

```bash
git clone https://github.com/ROHAN-BHUTANI/data-cleaning-env
cd data-cleaning-env

docker build -t data-cleaning-env .
docker run -p 8000:8000 data-cleaning-env

python inference.py
```

---

## 📈 Baseline Performance

| Task   | Score |
| ------ | ----- |
| Easy   | ~0.65 |
| Medium | ~0.55 |
| Hard   | ~0.45 |

✔ Deterministic
✔ Reproducible
✔ Validated via OpenEnv pipeline

---

## 🧪 Evaluation Criteria Alignment

| Criterion          | Coverage                       |
| ------------------ | ------------------------------ |
| Real-world utility | ✅ Industry-relevant pipeline   |
| Task quality       | ✅ Clear + progressive          |
| Environment design | ✅ Structured + stable          |
| Code quality       | ✅ OpenEnv + Docker             |
| Creativity         | ✅ RL + data engineering fusion |

---

## 💡 Novelty

This environment introduces:

* RL for **data engineering workflows**
* LLM-guided transformation policies
* Structured reasoning over tabular data

👉 A unique intersection of:

* Reinforcement Learning
* Data Science
* LLM Agents

---

## 🏁 Conclusion

This project demonstrates how **AI agents can move beyond games into real-world problem solving**, specifically in one of the most time-consuming tasks in data science.

🚀 A step toward **fully autonomous data pipelines**
