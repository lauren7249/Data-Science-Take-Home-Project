---
title: Exploratory Data Analysis
authors:
- Lauren Talbot
created_at: 2023-04-13 00:00:00
tldr: A machine learning project is only as good as the data that goes into it. What
  are some of the high level aspects of the data that we can discover? How should
  we clean and filter the data?
tags: []
updated_at: 2023-04-13 11:38:30.308340
---

# Raw Data Assessment & Processing


```python
import pandas
```
## Entity Relationships & Definitions


### Companies

do they share customers?

### Customers

do they share companies?

### Accounts

How do they relate to the above?

## Basic Data Processing

### Duplicate Keys

### Inaccurate Values

### Unclean Values

### Missing Values

### Inconsistencies

After merge

## Adding Analytical Variables 

### Date Quantity Variables

#### Invoice-Level

Days late: cleared_date - due date
Payment window: due_date - invoice_date
Days open: cleared_date - invoice_date

#### Broken Down By Period

What period-level should we use? (day, week, month)
Create periods from invoice date to close date
Rolling payment window: due_date - current period
Rolling days open: cleared_date - current period 

# Metadata Calculations & Cleaning

Totals, Uniques, Averages, Ranges, Outliers, Missings
Variables: Invoices, USD Amounts, Cleared/Open, Due Date, Invoice Date, Transaction Date, Customers, Companies, Accounts

# Notes

## Notable entities

e.g. customers with notable values

## Sparsity

### Entities

### Date Periods

## Trends Over Time

# Analysis

## Business Motivation

Cash collections don't follow due dates

## Data Science Benchmark

Define & Quantify: customers' mean absolute % error each period from cash due.

## Data Science Target

Best outcome variable? 
Days late
Days open (Total and Rolling)
Days Open as a % of Payment Window (Total and Rolling)
Days Late as a % of Payment Window (Total and Rolling)