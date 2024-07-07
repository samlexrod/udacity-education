# Bike Sharing Demand


## Overview 
In this project, I will be completing closed [Bike Sharing Competition](https://www.kaggle.com/c/bike-sharing-demand/overview) from Kaggle. 


The goal is to combine historical usage patterns with weather data in order to forcaste bike sharing demand in the Capital Bikeshare program in Washington, D.C.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
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


## Project Structure
The project structure is as follows:

```bash
bike-sharing-demand/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── eda.ipynb
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

Splitting modeling and evaluation notebooks, as well as separating code into a src directory, significantly enhances organization, clarity, and maintainability in a machine learning project. Separate notebooks allow clear focus on distinct phases, making debugging easier and improving collaboration, as team members can work on different tasks simultaneously. The src directory modularizes the code, facilitating independent modifications and reuse. This structure also improves reproducibility and documentation, as each component provides detailed, phase-specific insights. Overall, these practices lead to a more structured, efficient, and collaborative development process.

## Usage
To run the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/samlexrod/bike-sharing-demand.git
    cd bike-sharing-demand
    ```

2. Create an environment. I recommend using [Miniconda](https://docs.anaconda.com/miniconda/) if working on your machine.
    > Feel free to use a higher python version. Check versions [here](https://www.python.org/doc/versions/). I always use one minor version less to avoid bugs with newer versions.

    
    ### Create the virtual environment:
    ```bash
    conda create -n bikeshare python=3.11
    ```
    > If for any reason you need to remove the virtual environment and recreate, use `conda ramove --name bikeshare --all`

    ### Activate the virtual environment:
    ```bash
    conda activate bikeshare
    ```
    > Use `conda deactivate` to go back to base.

    ### Install the requirement-prior-autogluon.txt
    ```bash
    pip install -r requirements-prior-autogluon.txt
    ```
    > Use `pip list --format=freeze > requirements.txt` to freeze a new requirment file.

    ### Install Ydata Profiling
    ```bash
    conda install -c conda-forge ydata-profiling
    ```


    ### If using conda, install `autogluon` with mamba:
    >  WARNING! Install `mamba` on base environment.
    ```bash
    base> conda install -c conda-forge mamba
    base> conda activate bikeshare
    bikeshare> mamba install -c conda-forge autogluon
    ```
    > For more information about AutoGluon, visit ther website [here](https://auto.gluon.ai/0.8.1/install.html).

5. Run the notebooks in the notebooks/ directory to explore the data, build the model, and evaluate its performance.

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
