# Bike Sharing Demand


## Overview 
In this project, I will be completing closed [Bike Sharing Competition](https://www.kaggle.com/c/bike-sharing-demand/overview) from Kaggle. 


The goal is to combine historical usage patterns with weather data in order to forcaste bike sharing demand in the Capital Bikeshare program in Washington, D.C.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Introduction

Bike sharing systems are a means of renting bicycles where individuals can borrow a bike from one location and return it to a different location. These systems have gained popularity in urban areas due to their convenience and environmental benefits. This project aims to build a predictive model to forecast bike rental demand based on historical usage data.

## Dataset

The dataset used in this project contains historical data on bike rentals from the [Capital Bikeshare](https://www.capitalbikeshare.com/) system in Washington, D.C. It includes various features such as datetime, season, weather, temperature, humidity, and more.

The dataset can be downloaded from the Kaggle competition page [here](https://www.kaggle.com/competitions/bike-sharing-demand/data).

## Installation

To get started with this project, you need to have Python installed on your system. Additionally, you need to install the required libraries. You can do this by running:

```bash
pip install -r requirements.txt
```

Project Structure
The project structure is as follows:

> Note: At the moment this is just a draft of the structure of the project.

```bash
bike-sharing-demand/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── modeling.ipynb
│   └── evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluate.py
├── README.md
├── requirements.txt
└── LICENSE
```

Usage
To run the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/samlexrod/bike-sharing-demand.git
    cd bike-sharing-demand
    ```

2. Create an environment. I recommend using [Miniconda](https://docs.anaconda.com/miniconda/) if working on your machine.

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the notebooks in the notebooks/ directory to explore the data, build the model, and evaluate its performance.

## Modeling
The modeling process involves several steps:

- Data preprocessing: Handling missing values, encoding categorical variables, and scaling numerical features.
- Feature engineering: Creating new features based on domain knowledge and data exploration.
- Model selection: Trying different machine learning algorithms such as linear regression, decision trees, and gradient boosting.
Hyperparameter tuning: Optimizing the model parameters for better performance.


## Evaluation
The models are evaluated based on the Root Mean Squared Logarithmic Error (RMSLE) metric. The evaluation notebook provides a detailed analysis of the model performance on the validation set.

## Results
> No results yet.
