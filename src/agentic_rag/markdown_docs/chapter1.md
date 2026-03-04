# **What is a data** **warehouse?**

**DATA WAREHOUSING CONCEPTS**



**Aaren Stubberfield**
Data Scientist




--- end of page=0 ---

## **What you will learn**

What is a data warehouse


Warehouse architectures and properties


Data warehouse data modeling


Data prep and cleaning



**DATA WAREHOUSING CONCEPTS**




--- end of page=1 ---

## **What is a data warehouse?**

**A computer system designed to store and analyze large amounts of data for an**
**organization.**


**DATA WAREHOUSING CONCEPTS**




--- end of page=2 ---

## **What does a data warehouse do?**

Gathers data from different areas of an

organization


Integrates and stores the data


Make it available for analysis


1 Photos from Pexel by Nataliya Vaitkevich and Tiger Lily



**DATA WAREHOUSING CONCEPTS**




--- end of page=3 ---

## **Why is a data warehouse valuable?**

Organizations implement data warehouses in order to:*


Support business intelligence activity


Enable effective organizational analysis and decision-making


Find ways to innovate based on insights from their data


1 Data Management Book Of Knowledge 2nd Edition



**DATA WAREHOUSING CONCEPTS**




--- end of page=4 ---

## **Meet Bravo!**

Hypothetical publicly traded company

Sells home office furniture


1 Photo from Pexel by Pixabay



**DATA WAREHOUSING CONCEPTS**




--- end of page=5 ---

## **Common scenarios**

Product sales forecasting


Governance and regulation adherence


Insight and growth



**DATA WAREHOUSING CONCEPTS**




--- end of page=6 ---

## **Summary**

**What is a data warehouse?**


A computer system designed to store and
analyze large amounts of data for an
organization.


**What does a data warehouse do?**


Gathers data from different areas


Integrates and stores the data


Make it available for analysis



**Why is a data warehouse valuable?**


Support business intelligence activity


Enable effective analysis and decisionmaking


Foster data-driven innovation


**DATA WAREHOUSING CONCEPTS**




--- end of page=7 ---

# **Let's practice!**

**DATA WAREHOUSING CONCEPTS**




--- end of page=8 ---

# **What's the** **difference between**


# **data warehouses**


# **and data lakes?**



**DATA WAREHOUSING CONCEPTS**



**Aaren Stubberfield**
Data Scientist




--- end of page=9 ---

## **Database**

Structured data in rows and columns


Transactional databases store transactions



**DATA WAREHOUSING CONCEPTS**




--- end of page=10 ---

## **Data warehouse**

Gather data, integrate, and make available

for analysis


Many input data sources


Stores structured data


Complex to change

Upstream and downstream effects must
be considered


Typically >100 GB in size



**DATA WAREHOUSING CONCEPTS**




--- end of page=11 ---

## **Why the data warehouse?**

How quickly the query will run on a large

amount of data


Avoid slowing down transactional database



**DATA WAREHOUSING CONCEPTS**




--- end of page=12 ---

## **Data marts**

A relational database for analysis


Data is focused on one subject area


Few input data sources


Typically <100 GB in size



**DATA WAREHOUSING CONCEPTS**




--- end of page=13 ---

## **Data lake**

Entire organization store of data

Contains data from many departments


Many data input sources


Typically >100 GB in size


Stores structured and unstructured data

Examples: video, audio, and documents



**DATA WAREHOUSING CONCEPTS**




--- end of page=14 ---

## **Data lake**

Less complex to make changes

Fewer upstream and downstream effects
to consider


Purpose to store data may not be known

Less organized



**DATA WAREHOUSING CONCEPTS**




--- end of page=15 ---

## **Summary**

|Feature|Data Warehouse|Data Mart|Data Lake|
|---|---|---|---|
|Data structure|Structured|Structured|Structured &<br>Unstructured|
|Complexity to change|Complex|Complex|Less complex|
|Purpose of data|Known|Known|May not be known|
|Coverage of<br>departments|Covers many|Covers only<br>one|Covers many|
|Data sources|Many source<br>systems|Few sources|Many source systems|
|Typical size|>100 GB|<100 GB|>100 GB|



**DATA WAREHOUSING CONCEPTS**




--- end of page=16 ---

# **Let's practice!**

**DATA WAREHOUSING CONCEPTS**




--- end of page=17 ---

# **Data warehouses**


# **support** **organizational**


# **analysis**



**DATA WAREHOUSING CONCEPTS**



**Aaren Stubberfield**
Data Scientist




--- end of page=18 ---

## **High-level life cycle**



**DATA WAREHOUSING CONCEPTS**




--- end of page=19 ---

## **Planning - business requirements**

1. Business Requirements:

Understanding the organizational needs


Personas:

Analyst & Data Scientist - collect

requirements



**DATA WAREHOUSING CONCEPTS**




--- end of page=20 ---

## **Planning - data modeling**

1. Data Modeling:

Planning and organizing on integrating
data


Personas:

Data Engineer & Database Admins design data pipeline


Analyst & Data Scientist - data
business knowledge



**DATA WAREHOUSING CONCEPTS**




--- end of page=21 ---

## **Implementation - ETL Design & Development**

1. ETL Design:

Implement data pipelines and ETL
process


Personas:

Data Engineer & Database Admins implement data pipeline


**DATA WAREHOUSING CONCEPTS**




--- end of page=22 ---

## **Implementation - BI Application Development**

1. BI Application Development:

Setup business intelligence (BI) tools


Personas:

Analyst & Data Scientist - consult on
BI tool setup


**DATA WAREHOUSING CONCEPTS**




--- end of page=23 ---

## **Support / Maintenance - Maintenance**

1. Maintenance:

Make any needed modifications


Personas:

Data Engineer - modify as needed



**DATA WAREHOUSING CONCEPTS**




--- end of page=24 ---

## **Support / Maintenance - Test & Deploy**

1. Test & Deploy:

Testing


Personas:

Analyst & Data Scientist - consult on

BI tool setup


Data Engineers - deploy the data
warehouse


**DATA WAREHOUSING CONCEPTS**




--- end of page=25 ---

## **Persona matrix**







|Life cycle step|Analysts|Data<br>Scientist|Data<br>Engineers|Database<br>Administrators|
|---|---|---|---|---|
|Business Requirements|X|X|||
|Data Modeling|X|X|X|X|
|ETL Design &<br>Development|||X|X|
|BI Application<br>Development|X|X|||
|Maintenance|||X||
|Test & Deploy|X|X|X||


**DATA WAREHOUSING CONCEPTS**




--- end of page=26 ---

# **Let's practice!**

**DATA WAREHOUSING CONCEPTS**




--- end of page=27 ---

