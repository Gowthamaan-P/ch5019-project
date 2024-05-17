# CH5019 - Mathematical Foundations of Data Science

## Course Project

This repository/folder contains all files for the Course Project of CH5019 - Mathematical Foundations of Data Science course at IIT Madras.

### Github Repository

This project can be tracked using this Github Repository: https://github.com/Gowthamaan-P/ch5019-project

### Task

Solve the questions in the [Project Statements](Project%20Statements.pdf) file. There are two questions in this task:

1. Linear Regression
2. Unevenly Sampled Time Series Analysis

### Results

Results for both questions are in the [Q1 Result](Q1_RESULT.md) and [Q2 Result](Q2_RESULT.md) files. You can also read the detailed report here. The entire repository is organised like this,

``` graphql
Project
│   Project Statements.pdf     # Project statements
|   ED23S027_Report            # Project report
│   Q1_RESULT.md               # Results of Q1
│   Q2_RESULT.md               # Results of Q2
│   README.md                  # README file for the project
│
├── dataset                    # Contains all the dataset used in this project
│       medical_insurance.csv  # CSV file containing Medical insurance data (Q1)
│       tesla_stock_price.csv  # CSV file containing Tesla stock price data (Q2)
│
├── matlab                        # Contains all the MATLAB scripts for the project
│   ├── Q1                        # MATLAB Files for Question-1
│   │   ├── leastMedianSquares.m  # MATLAB script for least median squares
│   │   ├── leastTrimmedSquares.m # MATLAB script for least trimmed squares
│   │   ├── metrics.m             # MATLAB script for computing metrics
│   │   ├── ordinaryLeastSquares.m# MATLAB script for ordinary least squares
│   │   ├── plotresult.m          # MATLAB script for plotting results
│   │   ├── xsign.m               # MATLAB script for signum function
│   │   ├── Q1P2.mlx              # MATLAB Live Script for Q1 Part 2
│   │   └── Q1P3.mlx              # MATLAB Live Script for Q1 Part 3
│   │
│   └── Q2                        # MATLAB Files for Question-2
│       ├── lomb_scale_periodogram.m # MATLAB script for Lomb scale periodogram
│       ├── metrics.m                # MATLAB script for computing metrics
│       ├── Q2P1.mlx                 # MATLAB Live Script for Q2 Part 1
│       └── Q2P2.mlx                 # MATLAB Live Script for Q2 Part 2
│
└── report
    ├── Q1.pdf        # PDF report for Q1
    ├── Q1P2.pdf      # PDF report for Q1 Part 2
    ├── Q1P3.pdf      # PDF report for Q1 Part 3
    ├── Q2.pdf        # PDF report for Q2 
    ├── Q2P1.pdf      # PDF report for Q2 Part 1
    └── Q2P2.pdf      # PDF report for Q2 Part 2
```