📌 Price Optimization Project
Applying data science to optimize pricing strategies.

📄 Executive Summary:
Problem Statement
Ecommerces usually struggle and spend a lot of time and team efort to figure out the best prices for their products. Good ecommerces also have a decent amount of products coming in everyday into their website and its not that simple to keep pricing these rpoducts very accuratly without the proper tools. If you don't price your products correctly you might miss sales if your product gets too expensive or you might loose some money selling it for a price below the market. 

Data Science Opportunity
The idea with this price prediction model is to analyse historical products data and train a model that can predict th best prices for a specific product based on this product name, category, description and brand. We intend to do this work using NLP and a well executed preprocessing of our dataset extracting the most of our text features.

Poject Takeaways
Price optimization can significantly increase revenue by aligning prices with demand and product especification
A data-driven approach helps avoid guesswork and improves decision-making.
Implementing the right model ensures pricing strategies are adaptive to market trends.

🚧 Current Progress
This project is in the modeling and evaluation phase. So far, the following has been completed:

✅ Data cleaning and preprocessing

✅ Exploratory Data Analysis (EDA)

✅ Log transformation to normalize the target variable

✅ Text preprocessing using CountVectorizer and token prefixing for context

✅ Baseline modeling using Linear Regression, Random Forest, and XGBoost

✅ Model evaluation using metrics such as R², RMSE, and MAE

✅ Feature importance analysis from each model

Next steps:

🔄 Expand to larger training samples

🧠 Experiment with more advanced NLP techniques using SpaCy

⚙️ Hyperparameter tuning and model selection

📈 EDA Insights
Here are a few key visual insights discovered during the exploratory data analysis phase:

🎯 Target Variable Distribution (Before and After Log Transformation)

![Before Log](image/price_dist_beforeLog)
![After Log](image/price_dist_afterLog)

The price variable was highly skewed. A log transformation helped achieve a more normal distribution, improving model performance.

🧾 Most Frequent Tokens by Text Column

![After Log](image/top_15_tokens_desc.png)

The most common tokens across product descriptions, names, brands, and categories reveal strong signals, especially from brand and category fields.


📁 Organization
Repository Structure
bash
Copy
Edit
price_optimization_project/
│── data/             # Dataset links, processed data (small files)
│── model/            # Final trained model files (joblib, pickle)
│── notebooks/        # Jupyter notebooks with analysis & modeling
│── docs/             # Reports, presentations, and summaries
│── references/       # Research papers, tutorials, relevant materials
│── src/              # Python scripts refactored from notebooks
│── .gitignore        # Ignore unnecessary files (e.g., .DS_Store, large datasets)
│── conda.yml         # Conda environment setup file
│── README.md         # Project overview (this file)
│── LICENSE           # Project licensing information
📊 Dataset
This project uses real-world datasets to train the models.

📂 Data Source: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge

📜 Data Description:
Product pricing history
Sales transactions
External factors (seasonality, competition)


