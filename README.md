# event_driven_congestion
# ASTraM: Post-Event Learning Loop

## Overview

The **Post-Event Learning Loop** is an extension of the ASTraM (Adaptive Smart Traffic Management) framework that enables continuous learning from completed traffic events.

Traditional traffic management systems typically rely on operator experience and lack mechanisms to systematically learn from previous incidents. This module addresses that gap by introducing a feedback-driven pipeline that records predictions, captures actual outcomes, monitors performance, detects model drift, and triggers retraining when necessary.

---

## Objectives

This module aims to:

- Log predictions generated for incoming events.
- Record actual outcomes once events are resolved.
- Continuously evaluate model performance.
- Detect performance degradation over time.
- Trigger retraining when model drift is observed.
- Enable a self-improving traffic management system.

---

## Workflow

```text
Incoming Event
       ↓
Prediction Generation
       ↓
Prediction Logging
       ↓
Event Resolution
       ↓
Outcome Recording
       ↓
Performance Evaluation
       ↓
Drift Detection
       ↓
Retraining Decision
```

---

## Features

### 1. Prediction Logging

For every new event, the system stores:

- Predicted priority level
- Predicted rerouting requirement
- Predicted event duration
- Prediction confidence/probability
- Timestamp information

Predictions are stored in:

```text
pipeline_out/predictions_log.csv
```

---

### 2. Outcome Recording

After an event concludes, the actual outcomes can be recorded:

- Actual priority level
- Actual rerouting requirement
- Actual event duration

These outcomes are used to evaluate prediction quality.

---

### 3. Performance Evaluation

The module computes the following metrics:

| Task | Metrics |
|-------|---------|
| Priority Classification | Accuracy, ROC-AUC |
| Rerouting Classification | Accuracy, ROC-AUC |
| Duration Prediction | Mean Absolute Error (MAE) |

---

### 4. Drift Detection

Performance is continuously compared against previously stored baseline metrics.

Model drift is detected when classification metrics drop significantly relative to the baseline.

Default drift threshold:

```python
DRIFT_TOL = 0.05
```

This indicates that a retraining trigger occurs if performance falls by more than 5%.

---

### 5. Automatic Retraining Trigger

Retraining is initiated if either:

1. The number of newly labelled events exceeds:

```python
RETRAIN_AFTER_N = 200
```

OR

2. Model performance deteriorates beyond the specified drift threshold.

Retraining re-executes:

- `astram_clean.py`
- `astram_models.py`

to refresh the trained models.

---

## Generated Files

### Prediction Ledger

```text
pipeline_out/predictions_log.csv
```

Contains all predictions along with recorded outcomes.

---

### Baseline Metrics

```text
pipeline_out/model_baseline_metrics.json
```

Stores the performance metrics captured during the most recent training cycle.

---

## Dependencies

Install required packages:

```bash
pip install pandas numpy scikit-learn
```

---

## Usage

Run the module using:

```bash
python astram_feedback_loop.py
```

---

## Demonstration Pipeline

The demonstration script performs the following steps:

1. Loads previously trained ASTraM artifacts.
2. Simulates a stream of recent traffic events.
3. Generates and logs predictions.
4. Records actual outcomes.
5. Evaluates model performance.
6. Detects performance drift.
7. Determines whether retraining is required.

---

## Directory Structure

```text
project/
│
├── astram_feedback_loop.py
├── decisions.py
├── astram_clean.py
├── astram_models.py
│
└── pipeline_out/
    ├── predictions_log.csv
    └── model_baseline_metrics.json
```

---

## Current Limitations

> **Note:** Owing to project time constraints, this module is currently provided as a prototype implementation intended to demonstrate the feasibility of integrating a post-event learning mechanism into the ASTraM framework.

Some components may require additional refinement before deployment in a production environment.

Current limitations include:

- Complete end-to-end integration with all ASTraM modules has not been fully validated.
- Automatic retraining may require dataset-specific configuration depending on input formats.
- Certain edge cases and exception handling routines remain under development.
- Additional large-scale testing on real-world traffic datasets is required.

Despite these limitations, the module successfully demonstrates a continuous feedback architecture capable of supporting adaptive and self-improving traffic management.

---

## Future Enhancements

Potential future extensions include:

- Online and incremental learning.
- Advanced concept drift detection algorithms.
- Visualization dashboards for model monitoring.
- Automated scheduling of retraining jobs.
- Integration with real-time traffic sensor infrastructure.

---

## Conclusion

The Post-Event Learning Loop extends ASTraM by introducing a continuous feedback mechanism that enables the system to learn from past events, monitor its own performance, and adapt over time. This represents an important step towards building intelligent and adaptive urban traffic management systems.

---

## Remarks

Developed as an extension to the **ASTraM (Adaptive Smart Traffic Management)** framework for the Smart India Hackathon problem statement:

**Event-Driven Congestion (Planned & Unplanned)**

Focus Area:

> Leveraging historical and real-time data to forecast event-related traffic impact and recommend optimal manpower deployment, barricading strategies, and diversion plans.
