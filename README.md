# DRL-Based Path Planning for Greenhouse Climate Sensing

This repository contains the python implementation of the method described in the paper:

📄 **Optimizing the Trajectory of Agricultural Robots in Greenhouse Climatic Sensing with Deep Reinforcement Learning**  
Ashraf Sharifi, Sara Migliorini, Davide Quaglia – ICCAD 2024  
[IEEE Link](https://ieeexplore.ieee.org/document/10553772)

## 🌱 Project Overview

Greenhouse climate mapping is critical but costly with fixed sensors. This project proposes a **Deep Reinforcement Learning (DRL)** approach to optimize the movement of agricultural robots used as **mobile sensors**, reducing measurement time and improving the accuracy of temperature predictions.

### 🔍 Key Features

- **Deep Q-Learning** to dynamically select Points of Interest (PoIs) based on contextual importance.
- Integration with **virtual sensor models** (LSTM) to estimate temperatures at unvisited locations.
- Adaptive stop patterns to prioritize PoIs with high temperature variability.

## 🧠 Methodology

- **Agent**: Agricultural robot
- **States**: Current PoI + environmental context
- **Actions**: Next PoI to visit
- **Reward**: Temperature variation since last visit
- **Policy**: Learned with Q-network (3-layer MLP)

## 🧪 Dataset

- Real data collected from greenhouses in Verona, Italy (2021–2022)
- Includes environmental features, timestamps, and ground truth temperatures at 7 PoIs

> 📂 Dataset and pretrained models are available in this repository or upon request.




