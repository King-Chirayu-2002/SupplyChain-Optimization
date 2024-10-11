# Sustainable Supply Chain Optimization with Multi-Objective Model

This repository contains code and methodology for optimizing supply chain operations by minimizing **cost**, **time**, and **CO2 emissions**. The project leverages **machine learning**, specifically **unsupervised learning (K-means)**, and a **multi-objective optimization model** using **scikit-learn**, **TensorFlow**, and **Pandas** for data handling.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Objective Function](#objective-function)
- [Unsupervised Learning](#unsupervised-learning)
- [Visualization](#visualization)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Introduction
This project aims to optimize supply chain management by integrating sustainability metrics into decision-making. We built a **multi-objective optimization model** that minimizes:
- Cost of transportation
- Shipping time
- CO2 emissions

The model takes into account real-time operational data, creating a balance between these critical factors for sustainable decision-making.

## Installation
To use the code, you will need:
- Python 3.x
- Required libraries: 
  - Pandas
  - Numpy
  - TensorFlow
  - Scikit-learn
  - Matplotlib
  - Seaborn

## Methodology

### 1. Data Preprocessing:
   - We started by cleaning the dataset, removing unnecessary columns like customer personal information (names, email) and product-specific identifiers (product description, images).
   - Missing values in columns like `Customer Lname`, `Order Zipcode`, and `Product Description` were handled appropriately.
   - We calculated the **distance** from the warehouse to customer locations using the **Haversine formula** for latitude and longitude.

### 2. Feature Engineering:
   - **Distance Calculation:** 
     ```python
     def haversine(lat1, lon1, lat2, lon2):
         R = 6371  # Earth radius in kilometers
         lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
         dlat = lat2 - lat1
         dlon = lon2 - lon1
         a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
         c = 2 * np.arcsin(np.sqrt(a))
         return R * c
     ```
     The distance was calculated between the warehouse location and each customer.

   - **CO2 Emissions Estimation:**
     CO2 emissions were estimated based on transport modes (air, road, water) and distances using predefined emission factors:
     ```python
     emission_factors = {
         'Road/Water': 15,  # g CO2 per km per tonne for water/road transport
         'Air/Road': 500    # g CO2 per km per tonne for air transport
     }
     df['CO2 Emissions'] = df.apply(lambda row: emission_factors[row['Shipping Mode']] * row['Distance'], axis=1)
     ```

### 3. Multi-Objective Optimization:
   - The **objective function** was defined as a weighted sum of three parameters: **Cost**, **Shipping Time**, and **CO2 Emissions**:
     \[
     \text{Objective Function} = w_1 \times \text{Cost} + w_2 \times \text{Time} + w_3 \times \text{CO2 Emissions}
     \]
     We used TensorFlow and Scikit-learn to find the best weights for the objective function. Using clustering (K-means), we minimized the objective function to find the optimal supply chain operation parameters.

### 4. Unsupervised Learning:
   - **K-means clustering** was applied to group similar records based on the features: **Total Cost**, **Shipping Time**, and **CO2 Emissions**.
   - We calculated the **silhouette score** to assess the quality of the clusters:
     ```python
     from sklearn.metrics import silhouette_score
     silhouette_avg = silhouette_score(df[['Total Cost', 'Shipping Time', 'CO2 Emissions']], clusters)
     print(f'Silhouette Score: {silhouette_avg}')
     ```

### 5. Visualization:
   - A **3D scatter plot** was used to visualize the relationship between **Total Cost**, **Shipping Time**, and **CO2 Emissions** across different clusters:
     ```python
     import matplotlib.pyplot as plt
     from mpl_toolkits.mplot3d import Axes3D

     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     ax.scatter(df['Total Cost'], df['Shipping Time'], df['CO2 Emissions'], c=clusters)
     ax.set_xlabel('Total Cost')
     ax.set_ylabel('Shipping Time')
     ax.set_zlabel('CO2 Emissions')
     plt.show()
     ```

## Dataset
- **Columns Used**: Days for Shipping (Real), Days for Shipment (Scheduled), Shipping Mode, Benefit per Order, Sales per Customer, Delivery Status, Latitude, Longitude, Order City, CO2 Emissions, etc.
- The dataset was cleaned and preprocessed to retain only the columns relevant to the optimization task.

## Objective Function
The objective function balances the three goals:
1. **Cost**: Minimizing transportation and operational costs.
2. **Time**: Reducing delivery time for customer satisfaction.
3. **CO2 Emissions**: Promoting environmental sustainability.

Weights (w₁, w₂, w₃) are assigned to each goal based on priority, enabling flexible decision-making.

## Unsupervised Learning
We applied **K-means clustering** to identify patterns in the data:
- Clusters were created based on **Total Cost**, **Shipping Time**, and **CO2 Emissions**.
- The **silhouette score** of 0.365 indicates moderately good clustering.

## Visualization
- A **3D scatter plot** illustrates the clustering of records based on **Total Cost**, **Shipping Time**, and **CO2 Emissions**.
- The visualization helps in understanding trade-offs between these factors across different clusters.

## Results
- The optimal cluster minimized the objective function, providing the best balance between cost, time, and emissions.
- The **silhouette score** for the K-means clustering was **0.365**, indicating moderate clustering quality.
- The **bar chart** of costs across different clusters revealed that Cluster 0 had the lowest average cost.

## Usage
1. **Run the preprocessing and feature engineering steps** to clean the dataset and generate features.
2. **Set up the optimization** by defining the weights for Cost, Time, and CO2 emissions.
3. **Run the clustering** to identify the optimal operational parameters.
4. **Visualize the results** using the provided 3D scatter plots and bar charts.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

