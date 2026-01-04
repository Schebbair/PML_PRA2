# PRA2: Multi-Agent Reinforcement Learning

**Subject:** Paradigms of Machine Learning / AI

**Date:** January 2026

## General Description

This project implements and analyzes Multi-Agent Reinforcement Learning (MARL) algorithms in two distinct environments:
1.  **Prisoner's Dilemma:** A non-zero-sum matrix game.
2.  **Level-Based Foraging (LBF):** A Grid World environment featuring both cooperative and competitive tasks.

Two algorithms have been implemented and compared:
* **Independent Q-Learning (IQL):** A decentralized approach where agents learn selfishly.
* **Centralized Q-Learning (CQL):** A centralized approach that optimizes the joint reward (social welfare).

---

## Installation and Requirements

### 1. Python Dependencies
The project requires **Python 3.10**. It is highly recommended to use a virtual environment (Conda or venv).

Install the necessary libraries using the following command:

```bash 
pip install -r requirements.txt
```
## 2. System Dependencies (Linux / WSL)

For Part 2, rendering is required to generate video recordings. If you are running this on Linux or WSL (Windows Subsystem for Linux), you must install the following system libraries:

```bash
sudo apt-get update
sudo apt-get install -y python3-opengl libglu1-mesa freeglut3-dev xvfb
```
# Execution Instructions

## Part 1: Prisoner's Dilemma
Trains agents in the matrix game and generates convergence plots.

- **Train IQL:**

```bash
python train_iql.py
```
Output: Q-Values and Returns plots showing convergence to the Nash Equilibrium (Defect-Defect).

- **Train CQL:**

```bash
python train_cql.py
```
Output: Joint Return plot showing convergence to the Social Optimum (Cooperate-Cooperate).

## Part 2: Level-Based-Foraging (LBF)

Trains agents in a 5x5 grid to collect food. Generates MP4 videos and a comparison plot.

**Execution Command** (Linux / WSL): Use this command if running on a server or WSL without a physical display.

```bash
xvfb-run -a python train_lbf.py
```

**Execution Command (Native Windows / Mac with Display):**

```bash
python train_lbf.py
```
**Outputs:**

- `lbf_comparison.png`: Comparison plot of IQL vs. CQL in both competitive and cooperative modes.

- `videos_lbf/ folder`: Contains .mp4 videos of the trained agents solving the task.

# File Structure

- `iql.py`: Implementation of the IQL agent class.

- `cql.py`: Implementation of the CQL agent class.

- `train_iql.py`: Training script for Part 1 (IQL).

- `train_cql.py`: Training script for Part 1 (CQL).

- `train_lbf.py`: Training script for Part 2 (includes state wrapper and video recording).

- `matrix_game.py`: Definition of the Prisoner's Dilemma environment.

- `utils.py`: Visualization tools for Part 1.

- `requirements.txt`: List of dependencies.
