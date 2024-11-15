Product Clustering for Online Retail Store

Project Overview

This project involves clustering products for an online retail store using customer behavior and product attributes, such as category, price, and user rating. The primary goal is to group similar products together, allowing businesses to gain insights into product categories and enhance recommendations for users.

Features

 • Data Preprocessing: Includes data cleaning steps like handling missing values and encoding categorical data.
 • Multiple Clustering Techniques: Employs both K-Means and Hierarchical Clustering algorithms for product grouping.
 • Cluster Validation: Uses the Elbow Method to determine the optimal number of clusters, helping ensure effective segmentation.
 • Visualization: Visualizes clusters in 2D and 3D to provide a clear understanding of product groups based on attributes such as price, quantity, and customer country.

Dataset

The project uses the Online Retail dataset, which contains transactional information from a UK-based online store, including details such as customer country, product price, and purchase quantity.

Methodology

 1. Data Cleaning: Removed irrelevant columns like InvoiceDate, Description, and StockCode, dropped rows with missing values, and encoded categorical data.
 2. Normalization: Standardized features to improve clustering accuracy. Two different normalization approaches were tested, with Min-Max scaling proving to be most effective.
 3. Clustering Model: Experimented with a range of K values using the Elbow Method, identifying 4 as the optimal cluster count. Final clustering was performed using the K-Means algorithm with k-means++ initialization.
 4. Visualization: Clustered products were visualized in 2D and 3D, allowing analysis based on key features such as country, price, and quantity.