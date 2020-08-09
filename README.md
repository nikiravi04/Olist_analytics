# What is the Purpose of this App?

For every company, whether big or small, customer retention is a hard pill. It is a green signal that indicates a company is doing good or not. 
Filtering fruitful customers from the haystack is always better for a company’s resources. 

## 1) Customer Segmentation - RFE analysis

RFE stands for Recency, Frequency, and Engagement value each corresponding to some key customer trait. 
RFE analysis is a variation of the RFM (Recency, Frequency, Monetary) marketing model used to quantify customer behavior. 
The framework works by grouping customers based on how recently a customer has purchased (recency), how often (frequency), and by how much (monetary).
Basically we would want to segment customers into :

    - Low Value -> less activity, not frequent buyers/visitors or aliens, low revenue
    - Mid Value -> moderately active, moderate buyers/visitors or roamers, moderate income(not too high or low)
    - High Value -> high activity, frequent buyers/vistiors or fans, high revenue
        
## 2) Predicting repeat customers

The assumption made that predicting repeat customers based on observations made from their first purchases would reveal 
trends that could potentially help identify customers who are more likely to make repeat purchases. 
This could help Olist build a model that could be used in the future to predict whether or not a new customer is likely to make a repeat purchase and tailor their marketing efforts to the customer’s profile.

## 3) Predicting customer's next purchase day
Predicting customer's next purchase based on their buying patterns. This would also help Olist strategize their marketing actions by giving bundles offers, deals for their
customers so that they can buy more often from Olist
        
## Who can use it?
This App will provide all details required for the marketing team in just a click.It will help them understand the customer activities, behaviour and 
market conditions. Analyzing these trends will in turn help the company decide which customers to focus on in order to retain them and provide them favourable 
offers and promotions, to which customers and at what price & time. Thereby improving the company's business and revenue.


## Algorithms/Techniques/Metrics Used

    - K-means clustering 
    - Logistic Regression
    - Random Forest Classifier
    - RFM analysis
    - KFold Validation


## Appilcation 

Created a Streamlit application which shows exactly how we did the analysis and all the visualizations which could be used by the Olist team to better understand their customer's. The application is hosted on Heroku.    