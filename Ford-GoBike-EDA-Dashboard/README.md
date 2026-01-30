# 🚴 Ford GoBike Analytics Dashboard

##  Project Overview

This project focuses on analyzing **Ford GoBike trip data** using a full **Exploratory Data Analysis (EDA)** process, followed by building an **interactive analytics dashboard** using **Python Dash** and **Plotly**.

The goal of the project is to clean, explore, and visualize bike-sharing data in a clear and interactive way to extract meaningful insights about user behavior, trip patterns, and system usage.

---

##  Project Workflow

The project was completed in the following stages:

###  Exploratory Data Analysis (EDA)

* Understanding the structure of the dataset
* Inspecting data types and distributions
* Detecting missing values and outliers
* Analyzing key features such as:

  * Trip duration
  * Start time (hour, day, date)
  * User type
  * Gender and age

###  Data Cleaning & Preprocessing

* Handling missing and invalid values
* Converting date and time columns to proper formats
* Creating new derived features:

  * Trip duration (minutes)
  * Start hour
  * Day of the week
  * User age
* Filtering unrealistic values (e.g., invalid ages)

###  Dashboard Development (Dash)

* Building an interactive dashboard using **Dash**
* Visualizing insights with **Plotly** charts
* Designing a responsive and modern UI
* Adding dynamic filters for better exploration

---

##  Dashboard Preview

![Ford GoBike Dashboard](screenshots/dashboard.png)

> Interactive Ford GoBike Analytics Dashboard built using Dash & Plotly

---

##  Dashboard Features

The dashboard includes:

* **Key Metrics Cards**

  * Total trips
  * Average trip duration
  * Number of available bikes
  * Average daily trips

* **Interactive Filters**

  * User type (Subscriber / Customer)
  * Gender
  * Day of the week

* **Visualizations**

  * Daily trips over time (Line chart)
  * Hourly trip distribution (Bar chart)
  * User type distribution (Pie chart)
  * Trips by day of the week (Bar chart)
  * Trip duration distribution (Histogram)

All charts update dynamically based on selected filters.

---

##  Technologies Used

* **Python**
* **Pandas** – data manipulation & preprocessing
* **NumPy** – numerical operations
* **Plotly** – interactive visualizations
* **Dash** – web dashboard framework
* **HTML & CSS** – custom dashboard styling

---

##  How to Run the Project

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Install required libraries:

```bash
pip install dash pandas numpy plotly
```

3. Make sure the dataset file is available in the project directory:

```text
fordgobike-tripdataFor201902.csv
```

4. Run the dashboard:

```bash
python dashboard.py
```

5. Open your browser and go to:

```text
http://localhost:8050
```

---

##  Key Insights

* Peak usage occurs during morning and evening commuting hours
* Subscribers generate most of the trips
* Weekdays show higher activity compared to weekends
* Trip durations follow a right-skewed distribution

---

##  Future Improvements

* Add map-based visualizations for stations
* Deploy the dashboard online (Render / Railway / Heroku)
* Add predictive analytics (demand forecasting)
* Improve UI animations and accessibility

---

## Author

* **Developer**: Ahmed Morad
* **Version**: 1.0.0
* **Last Updated**: January 2026
* **Project Type**: Training 

---

## Acknowledgment

This project was built for learning and demonstrating skills in **data analysis, visualization, and dashboard development** using real-world datasets.

Feel free to fork, explore, and enhance the project 🙌
