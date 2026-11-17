# G114 | MP107 — ATM Cash Demand Prediction

## Project Overview
This project is part of the SCSE3040 MLOps Mini Project. We are building an end-to-end machine learning pipeline to predict daily ATM cash withdrawals for Bhopal Cooperative Bank.

## Problem Statement
The bank needs to estimate the cash required at each ATM for the next day. Accurate predictions can help reduce cash shortages and avoid keeping unnecessary idle cash.

## Machine Learning Task
- **Task:** Regression
- **Target column:** `withdrawal_thousand_rs`
- **Evaluation metric:** Mean Absolute Error (MAE)
- **Dataset location:** Bhopal ATM cash demand dataset provided for MP107

## Group Details
- **Group ID:** G114
- **Problem ID:** MP107
- **Faculty:** Pulkit Dwivedi
- **Batch:** 40

## Team Members
| Enrollment Number | Name |
|---|---|
| S24CSEU0096 | UTKARSHSINGH |
| S24CSEU0170 | VarunSharma |
| S24CSEU0208 | MRIDULBISHT |

## Planned MLOps Pipeline
1. Data exploration and quality checks
2. Data cleaning and preprocessing
3. Model training and experiment tracking using MLflow
4. Model registration and versioning
5. Prediction API using FastAPI
6. Containerization using Docker
7. Automated testing using GitHub Actions
8. Deployment configuration using Kubernetes
9. Data drift monitoring and model retraining
10. Final predictions on the holdout dataset

## Repository Structure
The repository will contain the source code, configuration files, tests, model artifacts, deployment files, reports, and prediction output required by the project.

## Status
Project implementation is in progress.
