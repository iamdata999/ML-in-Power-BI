# ML-in-Power-BI

## **Introduction: Leveraging Python and Machine Learning in Power BI for Demand Forecasting**  

In today's data-driven world, businesses rely on **advanced analytics and machine learning (ML)** to make strategic decisions. One powerful approach is integrating **Python within Power BI**, allowing analysts to perform complex data modeling and forecasting directly in their BI environment.  

This study focuses on using **Python-based Machine Learning (ML) models** in Power BI to predict future demand for **XLarge Bags** of avocados. By employing **time series forecasting techniques like ARIMA (AutoRegressive Integrated Moving Average)**, we analyze historical sales data and generate projections for the next five years.  

## **Why Use Python in Power BI?**  
- **Advanced Analytics**: Python enables the use of statistical and ML models beyond Power BI’s built-in tools.  
- **Automated Forecasting**: ML models like ARIMA help detect trends, seasonality, and patterns in data.  
- **Interactive Dashboards**: Predictions are visualized dynamically within Power BI for real-time decision-making.  

By combining **Python’s predictive capabilities with Power BI’s visualization tools**, businesses can enhance demand forecasting, optimize inventory, and make data-driven decisions with greater accuracy.


## **Background: Integrating Python and Machine Learning in Power BI**  

With the growing complexity of business data, organizations are increasingly turning to **Machine Learning (ML) and data analytics** to gain actionable insights. **Power BI**, a widely used business intelligence tool, allows users to integrate **Python scripts** to enhance analytics and forecasting capabilities.  

In this study, we explore the use of **Python in Power BI** for **time series forecasting**, specifically predicting the future demand for **XLarge Bags** of avocados. By leveraging **historical sales data**, we apply the **ARIMA (AutoRegressive Integrated Moving Average) model** to forecast trends and detect potential fluctuations in demand.  

## **Why Time Series Forecasting?**  
Time series forecasting is crucial in industries such as **retail, supply chain, and finance**, where understanding demand patterns helps in:  
- **Optimizing inventory management** to reduce waste and overstocking.  
- **Anticipating market trends** for better decision-making.  
- **Improving revenue forecasting** by predicting future sales behavior.  

## **Python and ARIMA in Power BI**  
Using **Python in Power BI**, we implement **ARIMA**, a widely used statistical model for forecasting. This model helps:  
- **Identify demand trends** over different years.  
- **Forecast future sales** based on historical data patterns.  
- **Enable interactive data visualization** within Power BI dashboards.  

By integrating **Python’s machine learning power** with **Power BI’s visualization capabilities**, businesses can achieve **data-driven forecasting**, leading to **better resource planning and strategic decision-making**. 

check report here: [Power BI ML](/Link/Power%20BI)


## **Tools Used**

To integrate **Python and Machine Learning** into Power BI for **time series forecasting**, we utilized the following tools and libraries:

## **1. Power BI**
- A **business intelligence tool** that allows users to create **interactive reports and dashboards**.
- Supports **Python scripting**, enabling advanced **data manipulation, machine learning, and forecasting**.

## **2. Python**
- A powerful programming language used for **data analysis, visualization, and machine learning**.
- Integrated within **Power BI** for **automated data processing and forecasting**.

## **3. Pandas**
- A **data manipulation** library used for **cleaning, organizing, and aggregating** historical avocado sales data.
- Helps in **structuring time-series data** for analysis.

## **4. NumPy**
- Used for **numerical computing and array manipulation**.
- Helps in handling **large datasets efficiently**.

## **5. Matplotlib**
- A **data visualization library** for generating **line charts** to compare historical and forecasted data.
- Plots **trends over time** to make insights more accessible.

## **6. Statsmodels (ARIMA Model)**
- **Statsmodels** is used to implement **ARIMA (AutoRegressive Integrated Moving Average)**, a statistical method for **time series forecasting**.
- ARIMA helps predict future demand based on **historical trends and seasonality**.

## **7. VS Code**
- **VS Code** is used for writing and testing Python scripts **before integrating with Power BI**.

By leveraging these tools, we successfully integrated **machine learning predictions** into **Power BI dashboards**, allowing businesses to make **data-driven decisions** on avocado sales trends.

## **The Analysis**

## **1. Data Preparation**
Before running **machine learning models** in **Power BI**, we first **loaded and preprocessed** the dataset using **Python**. The dataset, containing historical avocado sales, was structured with key columns such as:
- **Date**: Timestamp of the recorded sales.
- **AveragePrice**: The average price per avocado.
- **Total Volume**: The total quantity of avocados sold.
- **XLarge Bags**: Sales volume of extra-large avocado bags.
- **Year**: The year associated with each record.

Since **Power BI** requires well-structured data, we used **Pandas** to clean and aggregate the dataset.

---

## **2. Time Series Forecasting**
To analyze and forecast the future **demand for XLarge Bags**, we employed **ARIMA (AutoRegressive Integrated Moving Average)**, a widely used **time series forecasting model**. The steps included:

### **a. Aggregating Data by Year**
- Since **yearly trends** offer better insights, we grouped sales data by **year** to analyze long-term trends in **XLarge Bags** demand.

### **b. Training the ARIMA Model**
- We fitted an **ARIMA (2,1,2)** model using the `statsmodels` library.
- The model learned from past demand trends to generate **future predictions**.

### **c. Forecasting Future Demand**
- We forecasted the demand for **XLarge Bags** for the next **five years**.
- The model predicted whether the **sales volume would increase or decline over time**.

---

## **3. Data Visualization**
To enhance readability in **Power BI**, we visualized historical and predicted data:
- **Historical Trends**: A **line chart** plotted past demand for **XLarge Bags**.
- **Forecasted Demand**: The **ARIMA model’s predictions** were overlaid on the historical data to indicate future trends.

---

## **4. Insights & Business Impact**
- If predictions indicate **declining demand**, businesses may reconsider offering **XLarge Bags** or set a **minimum order requirement**.
- If demand **remains steady or grows**, businesses might explore **pricing strategies** to optimize sales.


This analysis, combined with **Power BI's interactive dashboards**, allows businesses to make **data-driven decisions on avocado sales trends**.  

## **Inside the work**

# **Explanation of the ARIMA Forecasting Plot**

This chart displays the forecasting of **'XLarge Bags'** using the **ARIMA model**, ensuring that the minimum order is **5 Bags**.

## **1️⃣ Train Data (Blue Line)**
- Represents the **historical data** used to train the ARIMA model.
- Significant spikes in the quantity of **'XLarge Bags'** indicate **fluctuating demand** over time.

## **2️⃣ Actual Test Data (Green Line)**
- The test dataset contains the **real values** for **'XLarge Bags'** that were not used during training.
- However, the test data is **either very small or missing**, which could mean:
  - There were **very few or zero** **'XLarge Bags'** in recent data.
  - The dataset might have a **large gap between training and test samples**.

## **3️⃣ ARIMA Forecast (Red Line)**
- The **ARIMA model** predicts **future values** based on past patterns.
- The forecasted values remain **flat and close to 5**, meaning the model predicts **very low future demand**.
- Since we enforced a **minimum order of 5 Bags**, the forecast **does not drop below this threshold**.

## **🔍 Possible Issues**
- The **sudden drop** in values after training suggests that **'XLarge Bags'** might have **insufficient data** in the later periods.
- The model might be **overfitting** to earlier fluctuations but **fails to detect strong trends for the future**.

![Arima ML](/Assets/Arima%20ML.png)

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = r"your dataset"
avocado = pd.read_csv(file_path)

# Ensure 'XLarge Bags' column is numeric
avocado['XLarge Bags'] = pd.to_numeric(avocado['XLarge Bags'], errors='coerce')

# Drop missing values (if any)
avocado.dropna(subset=['XLarge Bags'], inplace=True)

# Use index as the time series
avocado.reset_index(inplace=True)

# Split into training and testing
train_size = int(len(avocado) * 0.8)
train, test = avocado['XLarge Bags'][:train_size], avocado['XLarge Bags'][train_size:]

# Fit ARIMA Model
model = ARIMA(train, order=(5,1,0))  # (p,d,q) can be tuned
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Ensure minimum order of 5 Bags
forecast = np.maximum(forecast, 5)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(train, label="Train Data")
plt.plot(range(len(train), len(train) + len(test)), test, label="Actual Test Data", color='green')
plt.plot(range(len(train), len(train) + len(test)), forecast, label="Forecast (Min 5 Bags)", color='red')
plt.legend()
plt.xlabel("Index")
plt.ylabel("XLarge Bags")
plt.title("XLarge Bags Forecasting with ARIMA (Min Order = 5 Bags)")
plt.show()
```

# **Explanation of the Visualization**  

## **1. What This Chart Represents**  
- The graph compares the **XLarge Bags sales** (🔴 **red line**) with **Total Volume sales** (🔵 **blue line**) over time.  
- Both values are **normalized** (scaled between 0 and 1) for better comparison.  
- A **moving average smoothing technique** is applied to reduce noise and highlight trends.  

## **2. Key Observations**  
- The **spikes** in **XLarge Bags** (**red line**) indicate **occasional large demand** but **overall lower volume** compared to total sales.  
- The **Total Volume** (**blue line**) follows a **similar trend** but with **larger fluctuations**, meaning other bag sizes contribute **more** to the total volume.  
- There are periods where **XLarge Bags have little to no sales**, suggesting they are **not consistently in demand**.  

## **3. What This Means for Decision-Making**  
- **If the goal is to phase out XLarge Bags**, the **low demand** suggests it would have **minimal impact** on overall sales.  
- **If keeping them but enforcing a minimum order of 5 bags**, it may **not significantly affect sales** since demand is already **inconsistent**.  
- The **strong correlation at certain points** suggests that **XLarge Bags contribute at peak times** but are **not a primary driver of sales**.  

![MA ML](/Assets/MA%20ML.png)

```py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Load Data
file_path = r"your dataset"
avocado = pd.read_csv(file_path)

# Select relevant columns
avocado = avocado[['XLarge Bags', 'Total Volume']].dropna()

# Apply moving average to smooth the data
avocado['XLarge Bags Smooth'] = uniform_filter1d(avocado['XLarge Bags'], size=50)
avocado['Total Volume Smooth'] = uniform_filter1d(avocado['Total Volume'], size=50)

# Normalize for better comparison
avocado['Total Volume Normalized'] = avocado['Total Volume Smooth'] / avocado['Total Volume Smooth'].max()
avocado['XLarge Bags Normalized'] = avocado['XLarge Bags Smooth'] / avocado['XLarge Bags Smooth'].max()

# Plot
plt.figure(figsize=(12,6))
plt.plot(avocado.index, avocado['Total Volume Normalized'], color='blue', alpha=0.5, label='Total Volume (Normalized)')
plt.plot(avocado.index, avocado['XLarge Bags Normalized'], color='red', label='XLarge Bags (Normalized)')
plt.xlabel('Index')
plt.ylabel('Normalized Sales Volume')
plt.title('Smoothed XLarge Bags vs Total Volume Over Time')
plt.legend()
plt.show()
```

## Historical and Predicted Demand for XLarge Bags Over Time

### Blue Line (Historical Data)
- Shows actual sales of XLarge Bags from 2015 to 2018.
- Demand peaked in 2017 and then dropped significantly in 2018.

### Red Dashed Line (Forecast for Next 5 Years)
- The model predicts a decline in demand for 2019, but a gradual recovery from 2020 to 2022.
- The demand appears to peak again in 2021 and 2022 before declining in 2023.

### Key Takeaways
- **Short-Term Decline:** The demand for XLarge Bags dropped after 2017, possibly due to market shifts, pricing, or consumer preferences.
- **Recovery Period:** The forecast suggests demand will increase again from 2020 to 2022.
- **Uncertain Future:** After 2022, demand declines again, which could indicate an overall downward trend in future years.

![Time series Forcast](/Assets/Time%20Series%20Forcast.png)

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
file_path = r"your dataset"
avocado = pd.read_csv(file_path)

# Ensure 'year' and 'XLarge Bags' exist
if "year" not in avocado.columns or "XLarge Bags" not in avocado.columns:
    raise KeyError("Ensure your dataset contains 'year' and 'XLarge Bags' columns.")

# Aggregate data by year to see the trend
yearly_data = avocado.groupby("year")["XLarge Bags"].sum().reset_index()

# Train ARIMA model for forecasting
model = ARIMA(yearly_data["XLarge Bags"], order=(2,1,2))
model_fit = model.fit()

# Predict for the next 5 years
future_years = np.arange(yearly_data["year"].max() + 1, yearly_data["year"].max() + 6)
future_forecast = model_fit.forecast(steps=5)

# Plot historical vs forecasted values
plt.figure(figsize=(10,5))
plt.plot(yearly_data["year"], yearly_data["XLarge Bags"], marker='o', label="Historical Data", color="blue")
plt.plot(future_years, future_forecast, marker='o', label="Forecast (Next 5 Years)", color="red", linestyle="dashed")

plt.title("Future Demand Prediction for XLarge Bags")
plt.xlabel("Year")
plt.ylabel("Total XLarge Bags Ordered")
plt.legend()
plt.show()
```

## What I Learned  

### 1️⃣ Power BI and Python Integration  
- Using Python scripts within Power BI allows for advanced machine learning and forecasting.  
- Visualizing machine learning predictions inside Power BI enhances decision-making with real-time insights.  

### 2️⃣ ARIMA for Forecasting  
- The ARIMA model is effective for time-series forecasting, especially for predicting sales trends.  
- It requires careful tuning of parameters `(p,d,q)` to avoid overfitting or underfitting.  
- The model works best when the data follows a consistent trend without extreme fluctuations.  

### 3️⃣ XLarge Bags Demand Insights  
- The demand for XLarge Bags is inconsistent, with long periods of low or no orders.  
- Applying a minimum order quantity (5 bags) ensures some level of sales, but future demand is uncertain.  
- XLarge Bags have minor contributions to total volume, suggesting that their removal may not significantly impact overall sales.  

### 4️⃣ Data Cleaning and Preparation  
- Aggregating data by year helped reveal long-term trends in XLarge Bags demand.  
- Normalizing data enabled a clearer comparison of XLarge Bags against total sales volume.  
- Missing values and data gaps can impact forecasting accuracy, requiring careful handling before model training.  

### 5️⃣ Business Impact  
- If demand for XLarge Bags continues declining, discontinuation could be a viable option.  
- Introducing bulk discounts for XLarge Bags could encourage larger purchases and improve demand stability.  
- Power BI’s ability to integrate machine learning models enhances business intelligence for better strategic planning.  


## Conclusions  

### 📌 Key Findings  
- The ARIMA model was used to forecast the future demand for XLarge Bags, revealing a declining trend.  
- Demand for XLarge Bags has been inconsistent over the years, with periods of little to no sales.  
- Total sales volume is primarily driven by other bag sizes, making XLarge Bags a minor contributor.  

### 🔍 Business Implications  
- **Phasing Out XLarge Bags**: Since demand is low, discontinuing XLarge Bags may have minimal impact on overall sales.  
- **Setting a Minimum Order Quantity**: Enforcing a 5-bag minimum order ensures sales continue but may not significantly change demand.  
- **Alternative Strategies**: Instead of discontinuation, bulk discounts or promotional offers could be explored to improve demand.  

### 💡 Power BI & ML Impact  
- Integrating Python in Power BI enables advanced machine learning forecasting, making data-driven decisions easier.  
- ARIMA is a useful time-series forecasting model, but results depend on data quality and parameter tuning.  
- The combination of Power BI visuals and ML models provides powerful insights for better strategic planning.  

### 🔮 Future Considerations  
- Testing other ML models like XGBoost or LSTM to improve forecasting accuracy.  
- Exploring external factors (seasonality, promotions) that may impact demand for XLarge Bags.  
- Refining data preprocessing techniques to improve model predictions.  
