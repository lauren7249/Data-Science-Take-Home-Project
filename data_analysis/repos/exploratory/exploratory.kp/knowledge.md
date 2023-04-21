---
title: Exploratory Data Analysis
authors:
- Lauren Talbot
created_at: 2023-04-13 00:00:00
tldr: A machine learning project is only as good as the data that goes into it. What
  are some of the high level aspects of the data that we can discover? How should
  we clean and filter the data?
tags: []
updated_at: 2023-04-20 23:19:20.098386
thumbnail: images/output_24_1.png
---

```python
%matplotlib inline
figsize = (10,3)
```
# Input Datasets  


```python
import pandas, numpy
pandas.options.display.float_format = '{:,.4f}'.format
data_folder = '../data'
date_format='%Y-%m-%d' #truncate datetimes to dates
id_columns = ["id","company_id","invoice_id","account_id","customer_id"]
id_column_types = dict(zip(id_columns,[str] * len(id_columns)))
invoices = pandas.read_csv(data_folder + '/invoice.csv', na_values='inf', dtype=id_column_types,
                           parse_dates=['invoice_date', 'due_date', 'cleared_date'], date_format=date_format)
payments = pandas.read_csv(data_folder + '/invoice_payments.csv', na_values='inf', dtype=id_column_types,
                           parse_dates=['transaction_date'], date_format=date_format)
```
## Dataset Definitions & Relationships

We have two input datasets: invoices and their payments.
- Payments are amounts in time, which are directly mapped to companies. 
- Invoices can have multiple payments, but usually only have 1. 


```python
invoices.dtypes
```




    id                                  object
    due_date                    datetime64[ns]
    invoice_date                datetime64[ns]
    status                              object
    amount_inv                         float64
    currency                            object
    company_id                          object
    customer_id                         object
    account_id                          object
    cleared_date                datetime64[ns]
    root_exchange_rate_value           float64
    dtype: object




```python
payments.dtypes
```




    amount                             float64
    root_exchange_rate_value           float64
    transaction_date            datetime64[ns]
    invoice_id                          object
    company_id                          object
    converted_amount                   float64
    dtype: object




```python
#The join key will be invoice_id, so it must be unique (and it is).
invoices.id.value_counts(dropna=False).value_counts(dropna=False)\
.to_frame(name="ids").rename_axis('invoices_per_id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ids</th>
    </tr>
    <tr>
      <th>invoices_per_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>113085</td>
    </tr>
  </tbody>
</table>
</div>




```python
#all payments are represented in both datasets 
len(set(payments.invoice_id) - set(invoices.id))
```




    0




```python
#7% of invoices do not have payments yet
len(set(invoices.id) - set(payments.invoice_id))/invoices.__len__()
```




    0.07127382057744175




```python
#invoices usually have one payment but may have more
payments.invoice_id.value_counts(dropna=False).value_counts(dropna=False, normalize=True)\
.to_frame(name="invoices").rename_axis('payments_per_invoice')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>invoices</th>
    </tr>
    <tr>
      <th>payments_per_invoice</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.9419</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0548</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0006</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0001</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0001</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



## Entity Definitions & Relationships

- Company: business entity for which Tesorio is forecasting cash collected. There are only two. Each company collects using multiple currencies from multiple customers. 
- Account: **In this limited dataset, accounts and companies are synonymous, so we ignore accounts.**  
- Customer: metadata about an invoice which is specific to each company. Each customer can have multiple currencies.


```python
invoices.groupby("company_id")[["customer_id","currency"]].nunique()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>currency</th>
    </tr>
    <tr>
      <th>company_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>114</th>
      <td>4509</td>
      <td>15</td>
    </tr>
    <tr>
      <th>14</th>
      <td>546</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
invoices.groupby("customer_id").company_id.nunique().value_counts()\
.to_frame(name='customers').rename_axis('companies_per_customer')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customers</th>
    </tr>
    <tr>
      <th>companies_per_customer</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5055</td>
    </tr>
  </tbody>
</table>
</div>




```python
invoices.groupby(["customer_id"]).currency.nunique().value_counts()\
.to_frame(name='customers').rename_axis('currencies_per_customer')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customers</th>
    </tr>
    <tr>
      <th>currencies_per_customer</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4426</td>
    </tr>
    <tr>
      <th>2</th>
      <td>583</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
invoices.groupby("company_id").account_id.nunique().to_frame(name="unique_accounts")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unique_accounts</th>
    </tr>
    <tr>
      <th>company_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>114</th>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
invoices.groupby("account_id").company_id.nunique().value_counts()\
.to_frame(name='count').rename_axis('companies_per_account')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>companies_per_account</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Data Cleaning Needs

### Payments

Transaction data begins in 2011 and ends 2021-05-18. We will assume this is when the data was pulled. 


```python
payments.__len__()
```




    111623




```python
payment_stats = payments.describe(include='all')
payment_stats.loc['% populated'] = payment_stats.loc['count']/payments.__len__()
payment_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>root_exchange_rate_value</th>
      <th>transaction_date</th>
      <th>invoice_id</th>
      <th>company_id</th>
      <th>converted_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>111,622.0000</td>
      <td>111,623.0000</td>
      <td>111623</td>
      <td>111623</td>
      <td>111623</td>
      <td>111,622.0000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>105025</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>48171</td>
      <td>114</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>108124</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9,416.9806</td>
      <td>0.9684</td>
      <td>2018-03-04 09:52:41.445221376</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9,128.7160</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>0.0008</td>
      <td>2011-04-13 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4,078.6262</td>
      <td>1.0000</td>
      <td>2016-08-05 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3,180.9720</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9,332.6655</td>
      <td>1.0000</td>
      <td>2018-08-18 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8,819.6202</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14,651.4959</td>
      <td>1.0000</td>
      <td>2020-02-15 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14,612.8617</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19,999.8792</td>
      <td>3.2533</td>
      <td>2021-05-18 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61,209.4348</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6,015.3362</td>
      <td>0.2446</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6,438.8126</td>
    </tr>
    <tr>
      <th>% populated</th>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
last_transaction_date = payments.transaction_date.max()
first_transaction_date = payments.transaction_date.min()
first_transaction_date, last_transaction_date
```




    (Timestamp('2011-04-13 00:00:00'), Timestamp('2021-05-18 00:00:00'))




```python
#converted_amount is reliable
(((payments.amount * payments.root_exchange_rate_value) - payments.converted_amount).abs()).max()
```




    1.0913936421275139e-11




```python
payments[payments.amount.isnull()!=payments.converted_amount.isnull()].__len__()
```




    0




```python
payments.select_dtypes(include='float').hist(bins=50, figsize=figsize, layout=(1,3))
```




    array([[<Axes: title={'center': 'amount'}>,
            <Axes: title={'center': 'root_exchange_rate_value'}>,
            <Axes: title={'center': 'converted_amount'}>]], dtype=object)





![png](images/output_24_1.png)


### Invoices

Must become active within the date range of the transactions data to ensure completeness.


```python
#opened outside of payment data time period or after they were due - need to filter 
(invoices.loc[invoices.invoice_date>last_transaction_date].__len__(), 
invoices.loc[invoices.invoice_date<first_transaction_date].__len__(), 
invoices.loc[invoices.invoice_date.dt.to_period('M')>invoices.due_date.dt.to_period('M')].__len__())
```




    (79, 79, 14)




```python
invoices = invoices.loc[(invoices.invoice_date>=first_transaction_date) &
                        (invoices.invoice_date<=last_transaction_date) & 
                        (invoices.invoice_date.dt.to_period('M')<=invoices.due_date.dt.to_period('M'))]
```

```python
#to compare to payments. Are we holding the customer accountable to USD or their own currency?
invoices['converted_amount_inv'] = invoices.amount_inv * invoices.root_exchange_rate_value
```

```python
invoices_stats = invoices.describe(include='all')
invoices_stats.loc['% populated'] = invoices_stats.loc['count']/invoices.__len__()
invoices_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>due_date</th>
      <th>invoice_date</th>
      <th>status</th>
      <th>amount_inv</th>
      <th>currency</th>
      <th>company_id</th>
      <th>customer_id</th>
      <th>account_id</th>
      <th>cleared_date</th>
      <th>root_exchange_rate_value</th>
      <th>converted_amount_inv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>112888</td>
      <td>112888</td>
      <td>112888</td>
      <td>112888</td>
      <td>112,888.0000</td>
      <td>112888</td>
      <td>112888</td>
      <td>112888</td>
      <td>112888</td>
      <td>112888</td>
      <td>112,888.0000</td>
      <td>112,888.0000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>112888</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>18</td>
      <td>2</td>
      <td>5054</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CLEARED</td>
      <td>NaN</td>
      <td>USD</td>
      <td>114</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>109264</td>
      <td>NaN</td>
      <td>84998</td>
      <td>109695</td>
      <td>1786</td>
      <td>109695</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>2018-03-30 15:44:57.895259136</td>
      <td>2018-02-22 18:34:17.288640</td>
      <td>NaN</td>
      <td>10,026.6987</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-04-27 06:29:11.144497408</td>
      <td>0.9708</td>
      <td>9,742.3893</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>2011-04-13 00:00:00</td>
      <td>2011-04-13 00:00:00</td>
      <td>NaN</td>
      <td>0.0276</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011-05-15 00:00:00</td>
      <td>0.0008</td>
      <td>0.0080</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>2016-08-27 00:00:00</td>
      <td>2016-07-22 00:00:00</td>
      <td>NaN</td>
      <td>5,030.1056</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-09-16 00:00:00</td>
      <td>1.0000</td>
      <td>4,161.6325</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>2018-09-22 00:00:00</td>
      <td>2018-08-19 00:00:00</td>
      <td>NaN</td>
      <td>10,018.0136</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-10-13 00:00:00</td>
      <td>1.0000</td>
      <td>9,588.6353</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>2020-03-21 00:00:00</td>
      <td>2020-02-16 00:00:00</td>
      <td>NaN</td>
      <td>15,030.4018</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-04-21 00:00:00</td>
      <td>1.0000</td>
      <td>15,058.0828</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>2021-08-29 00:00:00</td>
      <td>2021-05-18 00:00:00</td>
      <td>NaN</td>
      <td>19,999.9749</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2022-01-01 00:00:00</td>
      <td>1.6816</td>
      <td>32,285.4757</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5,768.0500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.2461</td>
      <td>6,286.1231</td>
    </tr>
    <tr>
      <th>% populated</th>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
invoices['months_allowed'] = invoices.due_date.dt.to_period('M') - invoices.invoice_date.dt.to_period('M')
invoices.months_allowed = invoices.months_allowed.map(lambda m: m.n if not pandas.isnull(m) else None)
#almost all invoices are due immediately or within 3 months. filter out the rest
invoices.months_allowed.value_counts(normalize=True, dropna=False)
```




    months_allowed
    1    0.7645
    2    0.1808
    0    0.0320
    3    0.0199
    5    0.0009
    4    0.0008
    6    0.0003
    7    0.0002
    11   0.0001
    8    0.0001
    9    0.0001
    12   0.0001
    10   0.0001
    13   0.0000
    18   0.0000
    14   0.0000
    16   0.0000
    19   0.0000
    15   0.0000
    Name: proportion, dtype: float64




```python
invoices['months_billing'] = invoices.cleared_date.dt.to_period('M') - invoices.invoice_date.dt.to_period('M')
invoices.months_billing = invoices.months_billing.map(lambda m: m.n if not pandas.isnull(m) else None)
#almost all invoices are cleared within a year. filter out ones that cleared before they opened. will clip to 12.  
invoices.months_billing.value_counts(normalize=True, dropna=False)
```




    months_billing
     1    0.4507
     2    0.2654
     3    0.0810
     0    0.0718
     4    0.0324
     9    0.0213
     5    0.0186
     6    0.0116
     10   0.0086
     7    0.0083
     8    0.0081
     12   0.0067
     11   0.0045
    -1    0.0016
     13   0.0015
     14   0.0012
     15   0.0009
     16   0.0006
    -2    0.0006
     17   0.0005
    -4    0.0004
     18   0.0004
    -3    0.0004
     19   0.0003
     20   0.0002
     21   0.0002
    -5    0.0002
     24   0.0001
     22   0.0001
     25   0.0001
     23   0.0001
     26   0.0001
     28   0.0001
    -7    0.0001
    -11   0.0001
    -10   0.0001
    -9    0.0001
     31   0.0001
     27   0.0001
    -6    0.0001
    -8    0.0001
     30   0.0001
     29   0.0001
     32   0.0001
    -13   0.0000
    -15   0.0000
    -16   0.0000
    -14   0.0000
    -12   0.0000
    -22   0.0000
    -17   0.0000
     34   0.0000
    -18   0.0000
     40   0.0000
     38   0.0000
    -21   0.0000
    -32   0.0000
    -19   0.0000
     36   0.0000
     35   0.0000
    Name: proportion, dtype: float64



### Exchange Rate

Exchange rates vary for both payments and open invoices. Customers would expect to pay the amount they were originally invoiced in their own currency, not the USD amount originally invoiced. Therefore, we should use raw amounts to determine how much is paid vs due. 


```python
# USD is not is always 1 - it varies a lot
currency_ranges = invoices.groupby("currency").root_exchange_rate_value.describe(percentiles=[])
(currency_ranges['max']/currency_ranges['min']).sort_values().plot(kind='bar', title="Exchange Rate Spread Ratio")
```




    <Axes: title={'center': 'Exchange Rate Spread Ratio'}, xlabel='currency'>





![png](images/output_33_1.png)



```python
# 1.6% of USD invoices have an exchange rate unequal to 1
invoices_usd = invoices.query("currency=='USD'").copy()
invoices_usd['exchange_rate_is_1'] = invoices_usd['root_exchange_rate_value'] == 1
1 - invoices_usd.exchange_rate_is_1.mean()
```




    0.01562389703287137




```python
# USD exchange rate variations from 1 tend to be invoices which took longer to clear
# This suggests that the invoice exchange rate is "current state data." 
time_to_clear = invoices_usd.cleared_date - invoices_usd.invoice_date
invoices_usd['months_to_clear'] = time_to_clear.map(lambda t: round(t.days/30))
```

```python
invoices_usd.groupby("exchange_rate_is_1").months_to_clear.agg(['mean','count'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>exchange_rate_is_1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>2.9209</td>
      <td>1328</td>
    </tr>
    <tr>
      <th>True</th>
      <td>1.8937</td>
      <td>83670</td>
    </tr>
  </tbody>
</table>
</div>




```python
invoices_usd.groupby(invoices_usd.months_to_clear.clip(upper=13, lower=-1))\
.exchange_rate_is_1.mean().plot(title='% of USD Invoices With Exchange Rate Equal to 1', figsize=figsize)
```




    <Axes: title={'center': '% of USD Invoices With Exchange Rate Equal to 1'}, xlabel='months_to_clear'>





![png](images/output_37_1.png)


### Invoice status vs cleared date

All invoices have a date cleared. 
When an invoice is open, the date cleared is set to the future, and seems to be an assumed value. 


```python
invoices.loc[invoices.cleared_date.isnull()].__len__()
```




    0




```python
invoices.status.value_counts(normalize=True, dropna=False).to_frame(name="% of Invoices")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>% of Invoices</th>
    </tr>
    <tr>
      <th>status</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CLEARED</th>
      <td>0.9679</td>
    </tr>
    <tr>
      <th>OPEN</th>
      <td>0.0321</td>
    </tr>
  </tbody>
</table>
</div>




```python
invoices.loc[invoices.cleared_date.isnull() != (invoices.status == 'OPEN'),['status','cleared_date']]\
.value_counts(dropna=False)
```




    status  cleared_date
    OPEN    2022-01-01      3624
    Name: count, dtype: int64




```python
#all open invoices have the same cleared date, which is in the future relative to the latest transaction
invoices.loc[invoices.status == 'OPEN'].cleared_date.value_counts(dropna=False)
```




    cleared_date
    2022-01-01    3624
    Name: count, dtype: int64




```python
#all cleared invoices have a cleared date within the payments data window
invoices.loc[invoices.status == 'CLEARED', ['invoice_date','cleared_date']].agg(['min','max'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>invoice_date</th>
      <th>cleared_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>2011-04-13</td>
      <td>2011-05-15</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2021-05-12</td>
      <td>2021-05-18</td>
    </tr>
  </tbody>
</table>
</div>




```python
#open invoices are already active
invoices.loc[invoices.status == 'OPEN', ['invoice_date','due_date']].agg(['min','max'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>invoice_date</th>
      <th>due_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>2019-06-27</td>
      <td>2019-08-31</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2021-05-18</td>
      <td>2021-08-29</td>
    </tr>
  </tbody>
</table>
</div>



### Merging & Checking for Consistency

- No individual payments are more than their invoices. 
- Exchange rates vary across payments.
- Companies are consistent between payments and invoices, when payments are present. 
- Amounts make the most sense in their original currencies vs in USD


```python
payments['transaction_month'] = payments.transaction_date.dt.to_period('M')
invoice_payments = invoices.rename(columns={"id":"invoice_id","amount_inv":"amount",
                                            "converted_amount_inv":"converted_amount"})\
.merge(payments, on="invoice_id", how='left', suffixes=('_inv', '_pmt'))
```

```python
invoice_payments.invoice_id.nunique()
```




    112888




```python
duplicated_columns = [col.replace('_pmt','') for col in invoice_payments.columns if col.endswith('_pmt')]
for col in  duplicated_columns:
    inconsistent_rows = invoice_payments.loc[invoice_payments[col + '_pmt']!=invoice_payments[col + '_inv']]
    print(f"{col}: {inconsistent_rows.__len__()/invoice_payments.__len__()} inconsistent rows in merged dataset")
```
    amount: 0.18237593431151808 inconsistent rows in merged dataset
    root_exchange_rate_value: 0.3063537368275677 inconsistent rows in merged dataset
    company_id: 0.06675985369079206 inconsistent rows in merged dataset
    converted_amount: 0.38631322558234915 inconsistent rows in merged dataset



```python
invoice_payments.query("company_id_pmt!=company_id_inv").company_id_pmt.value_counts(dropna=False)
```




    company_id_pmt
    NaN    7976
    Name: count, dtype: int64




```python
invoice_payments.query("amount_pmt!=amount_inv")[['amount_pmt','amount_inv']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_pmt</th>
      <th>amount_inv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13,812.0000</td>
      <td>21,789.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5,019.0176</td>
      <td>9,964.7219</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5,890.4138</td>
      <td>5,769.7408</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>2.2108</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>94.5922</td>
      <td>4,943.9350</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2,087.8125</td>
      <td>9,941.0096</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9,185.5456</td>
      <td>14,932.1411</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19,989.9866</td>
      <td>19,999.9749</td>
    </tr>
  </tbody>
</table>
</div>




```python
#no payment is more than the invoice amount in the original currency
invoice_payments.loc[invoice_payments.amount_pmt>invoice_payments.amount_inv].__len__()
```




    0




```python
#converting to USD creates payments that are higher than invoice totals
invoice_payments.loc[invoice_payments.converted_amount_pmt>invoice_payments.converted_amount_inv].__len__()
```




    11061



### Business Questions for the Data


```python
invoice_payments['amount_pmt_pct'] = (invoice_payments.amount_pmt/invoice_payments.amount_inv)
```

```python
# Rougly 12% of payments are partial
(invoice_payments.amount_pmt_pct.dropna()<1).mean()
```




    0.12387888354739184




```python
invoice_payments.amount_pmt_pct\
.plot(kind="hist",bins=50, title="% of Invoice Collected with Payment", figsize=figsize)
```




    <Axes: title={'center': '% of Invoice Collected with Payment'}, ylabel='Frequency'>





![png](images/output_56_1.png)



```python
invoice_payments.sort_values(by=['invoice_id','transaction_date'], inplace=True)
```

```python
invoice_payments.groupby("invoice_id").amount_pmt_pct.cumsum()\
.plot(kind="hist",bins=50, title="Summed % of Invoice Collected with Payment", figsize=figsize)
```




    <Axes: title={'center': 'Summed % of Invoice Collected with Payment'}, ylabel='Frequency'>





![png](images/output_58_1.png)



```python
#small percent of payments represent overpayments
invoice_payments['pmt_pct_cum'] = invoice_payments.groupby("invoice_id").amount_pmt_pct.cumsum()
(invoice_payments.pmt_pct_cum>1).mean()
```




    0.006654223129912198




```python
#invoices with no transactions: use payments data end date as date of 0 amount 
invoice_payments.transaction_month = invoice_payments.transaction_month\
.fillna(last_transaction_date.to_period('M'))
```

```python
#there can be multiple transactions per month
invoice_payments.groupby(["invoice_id","transaction_month"]).transaction_date\
.count().value_counts(normalize=True).head()
```




    transaction_date
    1   0.8840
    0   0.0699
    2   0.0446
    3   0.0013
    4   0.0002
    Name: proportion, dtype: float64




```python
#the transactions are not duplicates
invoice_payments.groupby(["invoice_id","transaction_month"]).amount_pmt\
.nunique().value_counts(normalize=True).head()
```




    amount_pmt
    1   0.8843
    0   0.0700
    2   0.0444
    3   0.0012
    4   0.0001
    Name: proportion, dtype: float64




```python
#but they are almost always on the same day
invoice_payments.groupby(["invoice_id","transaction_month"]).transaction_date.nunique()\
.value_counts(normalize=True).head()
```




    transaction_date
    1   0.9269
    0   0.0699
    2   0.0030
    3   0.0001
    Name: proportion, dtype: float64




```python
payment_totals = invoices.set_index('id')
payment_totals['pmt_pct_cum'] = invoice_payments.groupby("invoice_id").pmt_pct_cum.max().fillna(0)
payment_totals['transaction_date_max'] = invoice_payments.groupby("invoice_id").transaction_date.max()
payment_totals['collected_date'] = invoice_payments.query("pmt_pct_cum>=1")\
.groupby("invoice_id").transaction_date.min()
```
#### Comparing invoice status and % collected 

- Invoices with cleared status can still have amounts remaining. 
- Invoices with open status are rarely collected. 


```python
# define invoice as collected if payments meet invoice amount in original currencies. 91% are collected
payment_totals['collected'] = payment_totals.collected_date.isnull()==False
payment_totals.collected.mean()
```




    0.9052157891007016




```python
#define cleared based on status. 97% are cleared
payment_totals['cleared'] = payment_totals.status=='CLEARED'
payment_totals.cleared.mean()
```




    0.9678973850187796




```python
#6% of invoices have a mismatch between collected and cleared 
(payment_totals.collected!=payment_totals.cleared).mean()
```




    0.06271702926794699




```python
#on average, 94% of cleared invoices are fully collected, compared to <1% of open ones
#cleared invoices have 96% of their amounts collected on average
payment_totals.groupby("status", as_index=False)[['collected','pmt_pct_cum']].mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>status</th>
      <th>collected</th>
      <th>pmt_pct_cum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CLEARED</td>
      <td>0.9352</td>
      <td>0.9573</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OPEN</td>
      <td>0.0006</td>
      <td>0.0063</td>
    </tr>
  </tbody>
</table>
</div>




```python
#67% of invoices that have not been collected are cleared nonetheless 
payment_totals.groupby("collected", as_index=False).cleared.mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>collected</th>
      <th>cleared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



#### Cleared Invoices

Cleared invoices may or may not be collected. If not collected, cleared invoices tend to be more overdue, suggesting that invoices must be cleared at some point.


```python
cleared_invoices = payment_totals.query("cleared == True").copy()
cleared_invoices['months_late'] = \
(cleared_invoices.collected_date.fillna(cleared_invoices.transaction_date_max).dt.to_period('M')\
- cleared_invoices.due_date.dt.to_period('M')).map(lambda m: m.n if not pandas.isnull(m) else None)
```

```python
cleared_invoices.groupby("collected").months_late.agg(['mean','min','max'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>collected</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>1.0919</td>
      <td>-34.0000</td>
      <td>25.0000</td>
    </tr>
    <tr>
      <th>True</th>
      <td>0.5461</td>
      <td>-23.0000</td>
      <td>28.0000</td>
    </tr>
  </tbody>
</table>
</div>



#### Comparing date cleared to date collected

Rarely, there can be a delay between the date an invoice is collected to when it is cleared. We will only forecast invoices when they are open AND not collected. 


```python
payment_totals['clear_delay_months'] = (payment_totals.cleared_date.dt.to_period('M') \
- payment_totals.collected_date.dt.to_period('M')).map(lambda m: m.n if not pandas.isnull(m) else None)
```

```python
payment_totals.__len__()
```




    112888




```python
payment_totals.clear_delay_months.describe(percentiles=[0.001,0.999])
```




    count   102,188.0000
    mean          0.1623
    std           1.3628
    min          -1.0000
    0.1%          0.0000
    50%           0.0000
    99.9%        20.0000
    max          39.0000
    Name: clear_delay_months, dtype: float64



# Structuring Data for Business Problem

- The model will handle OPEN invoices and classify how many months in the future they will be collected. 
- Define an invoice as open between its invoice date and date cleared or collected, whichever is first. 

## Creating transaction periods to model historical invoices

To model the data, we have to look the invoices in each prior period they were open and calculate when they are collected relative to that time. 

To ensure completeness, the periods we use for modeling must fall within the date range of the transactions data. Since we will forecast a year in advance, the forecast period must also be at a year prior to when the transactions data ends. 


```python
import numpy

def forecast_periods(invoice_date, last_billing_date):
    period_start = max(invoice_date,first_transaction_date.to_period('M'))
    period_end = min(last_billing_date,(last_transaction_date - pandas.DateOffset(years=1)).to_period('M'))
    return pandas.period_range(period_start, period_end)

payment_totals['last_forecast_date'] = payment_totals[['cleared_date','collected_date']].min(axis=1)
payment_totals['forecast_month'] = numpy.vectorize(forecast_periods)\
(payment_totals.invoice_date.dt.to_period('M'), payment_totals.last_forecast_date.dt.to_period('M'))
invoice_forecast_periods = payment_totals.reset_index().explode('forecast_month').dropna(subset=['forecast_month'])
invoice_forecast_periods.forecast_month.agg(['min','max'])
```




    min    2011-04
    max    2020-05
    Name: forecast_month, dtype: period[M]



## Live test cases: current open invoices

Invoices that we will predict after creating the model, without knowing the accuracy of the predictions. 
Per the instructions, we only predict collection dates for open invoices. 

Use the day after the payments data ends as the present date. 


```python
present_date = last_transaction_date + pandas.DateOffset(days=1)
open_invoices = invoices.query("status=='OPEN'").copy()
open_invoices['forecast_month'] = present_date.to_period('M')
```

```python
open_invoices.forecast_month.agg(['min','max','count'])
```




    min      2021-05
    max      2021-05
    count       3624
    Name: forecast_month, dtype: object



## Process inputs for model training and predictions

Months have lower kurtosis than periods. 


```python
def process_model_inputs(invoices_at_time_periods):
    raw_input_columns = ['id','invoice_date', 'months_allowed','amount_inv', 'converted_amount_inv',
                         'currency','company_id','customer_id','forecast_month']
    output_col = 'collected_date'
    if output_col in invoices_at_time_periods.columns:
        raw_input_columns += [output_col, 'cleared_date']
    output_df = invoices_at_time_periods[raw_input_columns]
    #remove months_allowed > 3
    output_df = output_df[output_df.months_allowed.between(0,3)]
    #only forecast when the invoice is active. 
    output_df = output_df[output_df.forecast_month>=output_df.invoice_date.dt.to_period('M')]
    output_df['months_billing'] = (output_df.forecast_month \
                                  - output_df.invoice_date.dt.to_period('M')).map(lambda m: m.n).clip(upper=12)
    output_df['months_late'] = output_df.months_billing - output_df.months_allowed
    output_df.forecast_month = output_df.forecast_month.dt.to_timestamp()
    return output_df

open_invoices_to_score = process_model_inputs(open_invoices)
invoices_periods_to_model = process_model_inputs(invoice_forecast_periods)
```

```python
open_invoices_to_score.describe(include='all', percentiles=[0.001,0.999]).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>min</th>
      <th>0.1%</th>
      <th>50%</th>
      <th>99.9%</th>
      <th>max</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>3611</td>
      <td>3611</td>
      <td>12</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>invoice_date</th>
      <td>3611</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-03-18 12:47:39.263361792</td>
      <td>2019-06-27 00:00:00</td>
      <td>2020-03-20 20:38:24</td>
      <td>2021-04-07 00:00:00</td>
      <td>2021-05-15 00:00:00</td>
      <td>2021-05-18 00:00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>months_allowed</th>
      <td>3,611.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.3262</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>3.0000</td>
      <td>3.0000</td>
      <td>0.6454</td>
    </tr>
    <tr>
      <th>amount_inv</th>
      <td>3,611.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10,007.8561</td>
      <td>2.2108</td>
      <td>9.2192</td>
      <td>9,927.5238</td>
      <td>19,981.8375</td>
      <td>19,989.6533</td>
      <td>5,766.4095</td>
    </tr>
    <tr>
      <th>converted_amount_inv</th>
      <td>3,611.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9,775.5033</td>
      <td>0.9371</td>
      <td>3.7824</td>
      <td>9,575.8003</td>
      <td>23,758.3755</td>
      <td>23,952.9480</td>
      <td>6,210.8811</td>
    </tr>
    <tr>
      <th>currency</th>
      <td>3611</td>
      <td>11</td>
      <td>USD</td>
      <td>2833</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>company_id</th>
      <td>3611</td>
      <td>2</td>
      <td>114</td>
      <td>3564</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>customer_id</th>
      <td>3611</td>
      <td>1153</td>
      <td>105</td>
      <td>68</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>forecast_month</th>
      <td>3611</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-05-01 00:00:00</td>
      <td>2021-05-01 00:00:00</td>
      <td>2021-05-01 00:00:00</td>
      <td>2021-05-01 00:00:00</td>
      <td>2021-05-01 00:00:00</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>months_billing</th>
      <td>3,611.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.8347</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>12.0000</td>
      <td>12.0000</td>
      <td>1.4287</td>
    </tr>
    <tr>
      <th>months_late</th>
      <td>3,611.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.5084</td>
      <td>-2.0000</td>
      <td>-2.0000</td>
      <td>0.0000</td>
      <td>10.3900</td>
      <td>12.0000</td>
      <td>1.2656</td>
    </tr>
  </tbody>
</table>
</div>



## Selecting prediction target

Predict months til collected relative to forecast date. 

Normalization:
- If the invoice isn't collected within the payments data time period, assume it's collected the day after, which we are using as the present date. 
- Clip collection period to 13 months, which is outside the forecast window.


```python
invoices_periods_to_model['months_til_collected'] = \
(invoices_periods_to_model.collected_date.dt.to_period('M') \
- invoices_periods_to_model.forecast_month.dt.to_period('M')).map(lambda m: m.n if not pandas.isnull(m) else None)
# why we clip outliers 
invoices_periods_to_model.months_til_collected.value_counts(normalize=True, dropna=False).head(20)
```




    months_til_collected
    0.0000    0.3280
    1.0000    0.3141
    2.0000    0.1539
    NaN       0.0695
    3.0000    0.0557
    4.0000    0.0276
    5.0000    0.0168
    6.0000    0.0108
    7.0000    0.0072
    8.0000    0.0049
    9.0000    0.0033
    10.0000   0.0024
    11.0000   0.0017
    12.0000   0.0012
    13.0000   0.0009
    14.0000   0.0006
    15.0000   0.0004
    16.0000   0.0003
    17.0000   0.0002
    18.0000   0.0002
    Name: proportion, dtype: float64




```python
#normalized values
invoices_periods_to_model['months_til_collected_norm'] = invoices_periods_to_model.months_til_collected.fillna(
    (present_date.to_period('M') - invoices_periods_to_model.forecast_month.dt.to_period('M')).map(lambda m: m.n)
).clip(upper=13)
invoices_periods_to_model.months_til_collected_norm.value_counts(normalize=True, dropna=False)
```




    months_til_collected_norm
    0.0000    0.3280
    1.0000    0.3141
    2.0000    0.1539
    13.0000   0.0714
    3.0000    0.0557
    4.0000    0.0276
    5.0000    0.0168
    6.0000    0.0108
    7.0000    0.0072
    8.0000    0.0049
    9.0000    0.0033
    10.0000   0.0024
    12.0000   0.0023
    11.0000   0.0017
    Name: proportion, dtype: float64




```python
invoices_periods_to_model.months_til_collected_norm.plot(kind='hist', bins=14, figsize=figsize, layout=(1,3), 
title="Months Til Collected, Up to 1 Year (collections 13+ months in the future are outside the forecast window)")
```




    <Axes: title={'center': 'Months Til Collected, Up to 1 Year (collections 13+ months in the future are outside the forecast window)'}, ylabel='Frequency'>





![png](images/output_90_1.png)



```python
invoices_periods_to_model.months_til_collected_norm.kurtosis()
```




    4.610682444381732




```python
invoices_periods_to_model['periods_til_collected'] = \
(invoices_periods_to_model.months_til_collected_norm/(invoices_periods_to_model.months_allowed+1)).clip(upper=13)

invoices_periods_to_model.periods_til_collected.plot(kind='hist', bins=14, figsize=figsize, layout=(1,3), 
      title="Billing Periods Til Collected (collections 13+ months in the future are outside the forecast window)")
```




    <Axes: title={'center': 'Billing Periods Til Collected (collections 13+ months in the future are outside the forecast window)'}, ylabel='Frequency'>





![png](images/output_92_1.png)



```python
invoices_periods_to_model.periods_til_collected.kurtosis()
```




    9.961900930076693




```python
invoices_periods_to_model.drop(columns=['collected_date','cleared_date','periods_til_collected'], 
                               inplace=True, errors='ignore')
invoices_periods_to_model.describe(include='all', percentiles=[0.001,0.999]).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>min</th>
      <th>0.1%</th>
      <th>50%</th>
      <th>99.9%</th>
      <th>max</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>250035</td>
      <td>90241</td>
      <td>34988</td>
      <td>37</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>invoice_date</th>
      <td>250035</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-04-02 19:06:22.741615616</td>
      <td>2011-04-13 00:00:00</td>
      <td>2011-05-18 00:00:00</td>
      <td>2017-09-03 00:00:00</td>
      <td>2020-05-26 00:00:00</td>
      <td>2020-05-31 00:00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>months_allowed</th>
      <td>250,035.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.2502</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>3.0000</td>
      <td>3.0000</td>
      <td>0.5318</td>
    </tr>
    <tr>
      <th>amount_inv</th>
      <td>250,035.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10,034.0164</td>
      <td>0.3583</td>
      <td>21.9605</td>
      <td>10,051.8132</td>
      <td>19,977.5079</td>
      <td>19,999.9749</td>
      <td>5,756.5608</td>
    </tr>
    <tr>
      <th>converted_amount_inv</th>
      <td>250,035.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9,759.1686</td>
      <td>0.0080</td>
      <td>3.3320</td>
      <td>9,617.8368</td>
      <td>26,261.1867</td>
      <td>32,285.4757</td>
      <td>6,316.3293</td>
    </tr>
    <tr>
      <th>currency</th>
      <td>250035</td>
      <td>17</td>
      <td>USD</td>
      <td>182090</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>company_id</th>
      <td>250035</td>
      <td>2</td>
      <td>114</td>
      <td>242048</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>customer_id</th>
      <td>250035</td>
      <td>4345</td>
      <td>7</td>
      <td>5056</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>forecast_month</th>
      <td>250035</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-04-28 03:17:52.808206848</td>
      <td>2011-04-01 00:00:00</td>
      <td>2011-06-01 00:00:00</td>
      <td>2017-10-01 00:00:00</td>
      <td>2020-05-01 00:00:00</td>
      <td>2020-05-01 00:00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>months_billing</th>
      <td>250,035.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.3775</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>12.0000</td>
      <td>12.0000</td>
      <td>1.8593</td>
    </tr>
    <tr>
      <th>months_late</th>
      <td>250,035.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1273</td>
      <td>-3.0000</td>
      <td>-3.0000</td>
      <td>0.0000</td>
      <td>11.0000</td>
      <td>12.0000</td>
      <td>1.8887</td>
    </tr>
    <tr>
      <th>months_til_collected</th>
      <td>232,656.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.3652</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>16.0000</td>
      <td>30.0000</td>
      <td>1.8678</td>
    </tr>
    <tr>
      <th>months_til_collected_norm</th>
      <td>250,035.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.1650</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>13.0000</td>
      <td>13.0000</td>
      <td>3.4296</td>
    </tr>
  </tbody>
</table>
</div>



## Analyze Input Data


```python
invoices_to_model = invoices_periods_to_model.query("months_billing==0").copy()\
.rename(columns={"forecast_month":"invoice_month"})
invoices_to_model['uncollected'] = invoices_to_model.months_til_collected.isnull()
```
### By Dates


```python
invoices_periods_to_model.groupby("forecast_month").id.count()\
.plot(kind='area', title="Invoices by Forecast Month", figsize=figsize)
```




    <Axes: title={'center': 'Invoices by Forecast Month'}, xlabel='forecast_month'>





![png](images/output_98_1.png)



```python
invoices_to_model.groupby("invoice_month").id.count()\
.plot(kind='area', title="Invoices by Invoice Month", figsize=figsize)
```




    <Axes: title={'center': 'Invoices by Invoice Month'}, xlabel='invoice_month'>





![png](images/output_99_1.png)


### By Currency

Some currencies have very low collection rates. This may be due to currency fluctuations.


```python
invoices_to_model.groupby("currency")\
.agg({"months_til_collected":["mean","std"],"id":"count","uncollected":"mean"})\
.sort_values(by=('uncollected','mean'), ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">months_til_collected</th>
      <th>id</th>
      <th>uncollected</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>currency</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TWD</th>
      <td>4.0000</td>
      <td>2.8284</td>
      <td>11</td>
      <td>0.8182</td>
    </tr>
    <tr>
      <th>HUF</th>
      <td>6.0000</td>
      <td>NaN</td>
      <td>2</td>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>HKD</th>
      <td>1.6667</td>
      <td>0.5164</td>
      <td>9</td>
      <td>0.3333</td>
    </tr>
    <tr>
      <th>INR</th>
      <td>2.6102</td>
      <td>2.4355</td>
      <td>84</td>
      <td>0.2976</td>
    </tr>
    <tr>
      <th>BRL</th>
      <td>2.1413</td>
      <td>1.8751</td>
      <td>465</td>
      <td>0.2237</td>
    </tr>
    <tr>
      <th>GBP</th>
      <td>1.9492</td>
      <td>2.0609</td>
      <td>1617</td>
      <td>0.1361</td>
    </tr>
    <tr>
      <th>SGD</th>
      <td>2.5217</td>
      <td>2.2274</td>
      <td>211</td>
      <td>0.1280</td>
    </tr>
    <tr>
      <th>KRW</th>
      <td>1.4675</td>
      <td>0.9260</td>
      <td>88</td>
      <td>0.1250</td>
    </tr>
    <tr>
      <th>CHF</th>
      <td>2.7241</td>
      <td>1.9253</td>
      <td>32</td>
      <td>0.0938</td>
    </tr>
    <tr>
      <th>CAD</th>
      <td>1.5238</td>
      <td>1.2498</td>
      <td>23</td>
      <td>0.0870</td>
    </tr>
    <tr>
      <th>EUR</th>
      <td>2.0152</td>
      <td>1.8452</td>
      <td>13956</td>
      <td>0.0739</td>
    </tr>
    <tr>
      <th>CNY</th>
      <td>2.5677</td>
      <td>2.2059</td>
      <td>1942</td>
      <td>0.0685</td>
    </tr>
    <tr>
      <th>AUD</th>
      <td>2.3349</td>
      <td>2.5023</td>
      <td>1115</td>
      <td>0.0682</td>
    </tr>
    <tr>
      <th>USD</th>
      <td>1.7371</td>
      <td>1.4888</td>
      <td>67547</td>
      <td>0.0562</td>
    </tr>
    <tr>
      <th>JPY</th>
      <td>1.5039</td>
      <td>0.9819</td>
      <td>3127</td>
      <td>0.0467</td>
    </tr>
    <tr>
      <th>RUB</th>
      <td>1.5455</td>
      <td>1.5076</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>NZD</th>
      <td>1.0000</td>
      <td>NaN</td>
      <td>1</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



### Trends Over Time

Invoice collection time and inability to collect have been trending down, which are good signs for Tesorio's business. 


```python
invoices_to_model.groupby("invoice_month").months_til_collected.mean()\
.plot(kind='line', title="Average Months to Collect by Invoice Month", figsize=figsize)
```




    <Axes: title={'center': 'Average Months to Collect by Invoice Month'}, xlabel='invoice_month'>





![png](images/output_103_1.png)



```python
invoices_to_model.groupby("invoice_month").uncollected.mean()\
.plot(kind='line', title="% Invoices Uncollected by Invoice Month", figsize=figsize)
```




    <Axes: title={'center': '% Invoices Uncollected by Invoice Month'}, xlabel='invoice_month'>





![png](images/output_104_1.png)


### By Customer 

We have trouble collecting from some customers, regardless of their currency. 


```python
customer_averages = invoices_to_model.set_index("customer_id").select_dtypes(include=['float','int','boolean'])\
.reset_index().groupby("customer_id").mean()
customer_averages.hist(bins=50, figsize=(10,5), layout=(2,4))
```




    array([[<Axes: title={'center': 'months_allowed'}>,
            <Axes: title={'center': 'amount_inv'}>,
            <Axes: title={'center': 'converted_amount_inv'}>,
            <Axes: title={'center': 'months_billing'}>],
           [<Axes: title={'center': 'months_late'}>,
            <Axes: title={'center': 'months_til_collected'}>,
            <Axes: title={'center': 'months_til_collected_norm'}>,
            <Axes: title={'center': 'uncollected'}>]], dtype=object)





![png](images/output_106_1.png)



```python
customer_stats = invoices_to_model.groupby("customer_id").uncollected.agg(['count','mean'])\
.add_prefix('uncollected_').sort_values(by="uncollected_mean", ascending=False)
customer_stats.query("uncollected_count>=30").uncollected_mean\
.plot(kind='hist', figsize=figsize, title="Customers with 30+ Invoices: % Uncollected", bins=50)
```




    <Axes: title={'center': 'Customers with 30+ Invoices: % Uncollected'}, ylabel='Frequency'>





![png](images/output_107_1.png)



```python
western_customer_stats = invoices_to_model.query("currency in ('USD','EUR','GBP')")\
.groupby("customer_id").uncollected.agg(['count','mean'])\
.add_prefix('uncollected_').sort_values(by="uncollected_mean", ascending=False)
western_customer_stats.query("uncollected_count>=30").uncollected_mean\
.plot(kind='hist', figsize=figsize, title="US and European Customers with 30+ Invoices: % Uncollected", bins=50)
```




    <Axes: title={'center': 'US and European Customers with 30+ Invoices: % Uncollected'}, ylabel='Frequency'>





![png](images/output_108_1.png)


# Business Analysis

## Business Motivation

Cash collections don't follow due dates. On average:

- 6% of total cash due each month is unpaid, equating to a \\$471K average deficit.
- Total cash collected each month is 9% off from the amount due, equating to a $571K average difference in cash flow. 


```python
amount_due = invoices\
.groupby(invoices.due_date.dt.to_period('M')).converted_amount_inv.sum().to_frame(name="amount_due_usd")
amount_paid = payments.rename(columns={"transaction_month":"due_month"})\
.groupby("due_month").converted_amount.sum().to_frame(name="amount_paid_usd")
business_motivation = amount_due.join(amount_paid, how='inner').reset_index(names='due_date')\
.query(f"due_date>'{first_transaction_date}' and due_date<'{last_transaction_date}'")
business_motivation['pct_unpaid'] = 1 - (business_motivation.amount_paid_usd/business_motivation.amount_due_usd)
business_motivation['unpaid'] = business_motivation.amount_due_usd - business_motivation.amount_paid_usd
business_motivation.set_index('due_date', inplace=True)
```

```python
business_motivation.pct_unpaid.plot(figsize=figsize, title="% Unpaid (USD Due)")
```




    <Axes: title={'center': '% Unpaid (USD Due)'}, xlabel='due_date'>





![png](images/output_112_1.png)



```python
business_motivation.pct_unpaid.mean(), business_motivation.pct_unpaid.abs().mean()
```




    (0.055369114130323256, 0.08972281867977448)




```python
business_motivation.unpaid.plot(figsize=figsize, title="USD Unpaid")
```




    <Axes: title={'center': 'USD Unpaid'}, xlabel='due_date'>





![png](images/output_114_1.png)



```python
business_motivation.unpaid.mean(), business_motivation.unpaid.abs().mean()
```




    (471494.2851179037, 570983.6203720493)



## Data Science Benchmark

Define & Quantify: customers' mean absolute % error each period from USD due.


```python
invoices_to_model['due_month'] = invoices_to_model.invoice_month.dt.to_period('M') \
+ invoices_to_model.months_allowed
```

```python
amount_due = invoices_to_model.groupby(["company_id","due_month"]).converted_amount_inv.sum()\
.to_frame(name="amount_due_usd")
amount_paid = payments.rename(columns={"transaction_month":"due_month","invoice_id":"id"})\
.merge(invoices_to_model[["id"]], on="id",how="inner").groupby(["company_id","due_month"]).converted_amount.sum()\
.to_frame(name="amount_paid_usd")
benchmark = amount_due.join(amount_paid, how='left').reset_index()\
.query(f"due_month>'{first_transaction_date}' and due_month<'{last_transaction_date}'")
benchmark['pct_unpaid'] = 1 - (benchmark.amount_paid_usd/benchmark.amount_due_usd)
benchmark['abs_pct_error'] = benchmark.pct_unpaid.abs()
```

```python
benchmark.groupby("company_id")[['pct_unpaid','abs_pct_error']].mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pct_unpaid</th>
      <th>abs_pct_error</th>
    </tr>
    <tr>
      <th>company_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>114</th>
      <td>-0.0887</td>
      <td>0.2440</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.0390</td>
      <td>0.3151</td>
    </tr>
  </tbody>
</table>
</div>




```python
benchmark[['pct_unpaid','abs_pct_error']].mean()
```




    pct_unpaid      -0.0662
    abs_pct_error    0.2763
    dtype: float64



##