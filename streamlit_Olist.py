#!/usr/bin/env python
# coding: utf-8



import markdown
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from PIL import Image
import requests
from io import BytesIO




def main():
    @st.cache
    def load_data(data_file):
        df = pd.read_csv(data_file)
        return df

    df_Olist_ecommerce = load_data('Olist_ecommerce_master.csv')
    df_user = load_data('data_streamlit/Olist_RFM.csv')
    df_marketing = load_data('data_streamlit/Olist_marketing.csv')
    df_seller_merged = load_data('data_streamlit/Olist_seller_merged.csv')
    df_pred_lr_model = load_data('data_streamlit/Olist_LR_pred.csv')
    df_pred_purchase = load_data('data_streamlit/Olist_predict_purchase.csv')
    df_pred_rf_model = load_data('data_streamlit/Olist_RF_pred.csv')

    st.sidebar.title("Olist's Marketing Analysis Application")
    genre = st.sidebar.radio(
      '''
      Choose an option
      ''',
     ('Home','EDA of Olist','Customer Segmentation','Predicting Repeat Customers', 'Predicting Next Purchase'))

    if genre == 'Home':
        st.markdown(

        '''

        # What is the Purpose of this App?

        For every company, whether big or small, customer retention is a hard pill. It is a green signal that indicates a company is doing good or not. 
        Filtering fruitful customers from the haystack is always better for a company’s resources. 

        ## 1) Customer Segmentation - RFE analysis

        RFE stands for Recency, Frequency, and Engagement value each corresponding to some key customer trait. 
        RFE analysis is a variation of the RFM (Recency, Frequency, Monetary) marketing model used to quantify customer behavior. 
        The framework works by grouping customers based on how recently a customer has purchased (recency), how often (frequency), and by how much (monetary).
        Basically we would want to segment customers into :

            Low Value -> less activity, not frequent buyers/visitors or aliens, low revenue
            Mid Value -> moderately active, moderate buyers/visitors or roamers, moderate income(not too high or low)
            High Value -> high activity, frequent buyers/vistiors or fans, high revenue
        
        ## 2) Predicting repeat customers
        The assumption made that predicting repeat customers based on observations made from their first purchases would reveal 
        trends that could potentially help identify customers who are more likely to make repeat purchases. 
        This could help Olist build a model that could be used in the future to predict whether or not a new 
        customer is likely to make a repeat purchase and tailor their marketing efforts to the customer’s profile.

        ## 3) Predicting customer's next purchase day
        Predicting customer's next purchase based on their buying patterns. This would also help Olist strategize their marketing actions by giving bundles offers, deals for their
        customers so that they can buy more often from Olist
        
        ## Who can use it?
        This App will provide all details required for the marketing team in just a click.It will help them understand the customer activities, behaviour and 
        market conditions. Analyzing these trends will in turn help the company decide which customers to focus on in order to retain them and provide them favourable 
        offers and promotions, to which customers and at what price & time.
        Thereby improving the company's business and revenue.


        ## Algorithms/Techniques/Metrics Used

            - K-means clustering 
            - Logistic Regression
            - Random Forest Classifier
            - RFM analysis
            - KFold Validation

        ## Accuracy for models

            - Logistic Regression - 92%
            - Random Forest Classifier - 97%

        ## Appilcation 

        Created a Streamlit application which shows exactly how we did the analysis and all the visualizations 
        which could be used by the Olist team to better understand their customer's. The application is hosted on Heroku.     
        
    
        ''')


        st.image('images/Olist_ecommerce.png',width=600)
        st.image('images/Olist_marketing.png',width=600)

    elif genre == 'EDA of Olist':
        st.title('Exploratory Data Analysis of Olist')
        st.write('''The EDA will be focussed more on customers and how their shopping patterns are. We explore those datapoints as we want to see how the customers 
            from Olist can be further segmented and what price range can be assigned to them to improve their shopping experience and in turn 
            increase the customer LTV(life time value) on the website ''')

        st.write('Total customers in Olist for the years 2016-2018')
        total_customers = df_Olist_ecommerce.customer_unique_id.nunique()
        st.write(total_customers)

        st.write('Total orders for each year for Olist')
        total_orders = df_Olist_ecommerce.groupby(['order_purchase_year']).order_id.nunique()
        total_orders.columns = ['Purchase Year','Count']
        st.write(total_orders)

        #creating monthly active customers dataframe by counting unique Customer IDs

        df_monthly_active = df_Olist_ecommerce.groupby('order_purchase_mon')['customer_unique_id'].nunique().reset_index()
        df_monthly_active = df_monthly_active.sort_values(['customer_unique_id','order_purchase_mon'],ascending=False)
        fig_mac = go.Figure(data=go.Scatter(x=df_monthly_active['order_purchase_mon'],
                                y=df_monthly_active['customer_unique_id'],
                                mode='lines')) # hover text goes here
        fig_mac.update_layout(title='Monthly Active Customers for Olist',xaxis_title="Month(1-12)",yaxis_title="Number of customers")
        st.plotly_chart(fig_mac, use_container_width=True)

        st.write("From the above plot we can see that, there is a varying manner on how many customers are actively purchasing on Olist."
            "We plot this data to see how our customer segementation & repeat customer's can help categorise customers and perform targetted marketing actions."
            "As we can see customers are not active during September & October months for the 2 years of data we have. If we divide them,  \n"
            "- Olist have more active customers during the first half of the year with August being the most active month and  \n"
            "- Olist have a decline in active customers post August during the second half of the year  \n")


        #top categories vs sale per year
        df_sales_per_category = df_Olist_ecommerce.groupby(['order_purchase_year', 'product_category_name'], as_index=False).payment_value.sum()
        df_sales_per_category = df_sales_per_category.sort_values(by=['payment_value'], ascending=False)
        df_sales_per_category.columns = ['Purchase Year','Product Category', 'Sales Revenue']

        fig_top_cat_sales = px.bar(df_sales_per_category, y='Sales Revenue', x='Product Category', text='Sales Revenue', hover_data=['Purchase Year'])
        fig_top_cat_sales.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_top_cat_sales.update_layout(barmode='stack',uniformtext_minsize=8, uniformtext_mode='hide',height=800)
        fig_top_cat_sales.update_layout(title='Top categories vs Sales Revenue (monthly)')
        st.plotly_chart(fig_top_cat_sales, use_container_width=True)

        st.write("From the above graph we can see, how the revenue for each category is doing for each year.  \n"
                   "- Bed bath table surprisingly has the most revenue  \n"
                    "- Followed by Health and Beauty category  \n"
                    "- And thrid being Computer Accessories  \n"

                    "This graph helps us understand what kind of product and from which categories do customers buy most often. "
                    "This could be a factor while predicting the next purchase for the customer")


        #Top categories for year 2018 in Olist
        df = df_Olist_ecommerce[df_Olist_ecommerce.order_purchase_year == 2018]
        sales_per_category = df.groupby(['product_category_name'], as_index=False).payment_value.sum()
        sales_per_category = sales_per_category.sort_values(by=['payment_value'], ascending=False)
        sales_per_category.columns = ['Product Category', 'Sales Revenue']

        sales_per_category = sales_per_category[:20]
        labels = sales_per_category['Product Category']
        values = sales_per_category['Sales Revenue']

        fig_top_cat_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig_top_cat_pie.update_layout(title='Top categories vs Sales Revenue for the year 2018')
        st.plotly_chart(fig_top_cat_pie, use_container_width=True)

        #total sales vs year
        total_rev_month = df_Olist_ecommerce.groupby(['order_purchase_year'], as_index=False).payment_value.sum()
        total_rev_month.columns = ['Sales Year', 'Sales Revenue']
        fig_total_sales_year = px.line(total_rev_month, x='Sales Year', y='Sales Revenue')
        fig_total_sales_year.update_layout(title='Total Sales per Year',xaxis_title="Year",yaxis_title="Sales Revenue")
        fig_total_sales_year.update_xaxes(nticks=3)
        st.plotly_chart(fig_total_sales_year, use_container_width=True)

        st.write('''Revenue seems to be increasing from the year 2016 to 2018. Upward trend. Great marketing stretegies like customer 
            segmentation and predictive repeting customers will help Olist ''')




    elif genre == 'Customer Segmentation':
        st.title('Customer Segmentation for Olist - RFM analysis')

        st.markdown('''
            ## K-mean clustering for RFM 

            The K-means algorithm was used as it is a popular unsupervised machine learning algorithm used to perform clustering and segmentation tasks. 
            As a brief overview, K-means groups similar data points together and looks for a fixed number (k) of clusters in the data. 
            It does this by first randomly generating k number of centroids to initialize the clusters. 
            Then it goes through a number of iterations where each data point is assigned to its nearest centroid based on the squared Euclidean distance. 
            Next, the centroids are updated by computing the mean of all data points assigned to that centroid's cluster. 
            The algorithm stops when the sum of distances are minimized or when a max number of iterations are reached.''')

        st.write(df_user.head(10))

        st.markdown('## Detailed Analysis of Customer Segmentation with Marketing Action')
        #calculating average values for each customer Segment, and return a size of each segment 
        rfm_level_agg = df_user.groupby('Detailed Customer Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count'],
            'Marketing Action': 'unique'
        }).round(1)

        rfm_level_ag = pd.DataFrame(rfm_level_agg)
        rfm_level_ag = rfm_level_ag.reset_index()
        st.write(rfm_level_ag)

        #plot for customer segmentation
        fig_rfm =go.Figure(go.Bar(
            x = rfm_level_ag['Detailed Customer Segment'], 
            y= rfm_level_ag[('Monetary', 'count')]
        ))


        fig_rfm.update_layout(title="Customer Segmentation based on RFM scores",height=600,width=1000)
        fig_rfm.update_yaxes(nticks = 5)

        st.plotly_chart(fig_rfm, use_container_width=True)

        st.markdown('## Importance of RFM among clusters')
        st.write("We can see that our grouped summary of the mean of R, F, M that each cluster of customers places a different emphasis on our 4 clusters:  \n"
                    "### Cluster 0  \n"
                    "It has the lowest Montary Value mean and high Recency mean and the lowest frequency mean which means no spending customers and most inactive customers — We will need to do something before we lose them!  \n"
                    "### Cluster 1  \n"
                    "It performs poorly across R, F, and M. we will need to design campaigns to activate them again  \n"
                    "### Cluster 2  \n"
                    "They shopped with us recently but have not spend as much or as frequently as we would like them to — perhaps some personalization of products targeted at them can help to maximize their lifetime-value and come back to purchase?  \n"
                    "### Cluster 3  \n"
                    "Highest Monetary value,High Frequency and Low Recency which means high spending ,active customers in this cluster — This is our ideal customer segment  \n")



    elif genre == 'Predicting Repeat Customers':
        st.title('Predicting Repeat Customers')

        st.write("The assumption made that predicting repeat customers based on observations made from their first purchases would reveal trends that "
            "could potentially help identify customers who are more likely to make repeat purchases. This could help Olist build a model that could be used in the "
            "future to predict whether or not a new customer is likely to make a repeat purchase and tailor their marketing efforts to the customer’s profile.")

        st.write(df_seller_merged.head())

        st.write("In preparation for this classification task, we started off by identifying customers" 
            "who have made repeat purchases using the boolean 1 or True, and 0 for non repeats (False)")

        st.image('images/LR_pred.png',width=500)

        st.write("From the classification report we can see that the predicitions are really good with a balanced dataset. It predicts well based on the recall and precision.  \n"

                "### Precision — What percent of your predictions were correct?  \n"
                "Precision is the ability of a classifier not to label an instance positive that is actually negative. For each class,"
                "it is defined as the ratio of true positives to the sum of a true positive and false positive.  \n"
                "Precision:- Accuracy of positive predictions.  \n"

                "In the 2 categories, the precision is high which is good  \n"

                "### Recall — What percent of the positive cases did you catch?  \n"
                "Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true "
                "positives to the sum of true positives and false negatives.  \n"
                "Recall:- Fraction of positives that were correctly identified.  \n"

                "In the category of whether the customer will be a repeat customer which is 1 for yes and 0 for no, we have a high recall for 0 and a recall of 41% for 1  \n"

                "### F1 score — What percent of positive predictions were correct?  \n"
                "The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. "
                "F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, "
                "the weighted average of F1 should be used to compare classifier models, not global accuracy.  \n"
                "We have a high F1 score which is 91% and which is great for the model")

        st.write(df_pred_lr_model.tail(10))

        df_user.loc[df_user['customer_unique_id'] == '5c58de6fb80e93396e2f35642666b693']

        st.write(" We can see that the probability of predecting a repeat customer is on point  \n"

                    "For example, below we took a customer_unique_id and we can see how we categorised the customer as a repeat "
                    "customer and the customer actually falls into the Mid-Value segment which needs attention"

                    "From this we can understand that, we need to offer Price incentives & limited offers to "
                    "repeat customers so that they can come back to Olist and purchase")

        

    elif genre == 'Predicting Next Purchase':
        st.title('Predicting Next Purchase by categorising customers based on the days')

        st.write("We will be using customers to purchase data to predict their future repurchase chance within a given period of time. "
            "We will be needing their purchase date, item price, review response time and customer id")

        st.write(df_pred_purchase.head())

        st.write(" We create a dataframe above which creates MinPurchaseDate, MaxPurchaseDate, and NextPurchaseDay. We then segment the customer based on the NextPurchaseDay columns")
        st.write("Segment customers based on the NextPurchaseDay  \n"
                    "### less than 1 month  \n"
                    "df_predict_next['NextPurchaseDayRange'] = 1    \n"
                    "###  more than 1 month  \n"
                    "df_predict_next.loc[df_predict_next.NextPurchaseDay > 30,'NextPurchaseDayRange'] = 2  \n"
                    "### more than 3 months  \n"
                    "df_predict_next.loc[df_predict_next.NextPurchaseDay > 90,'NextPurchaseDayRange'] = 3   \n")

        st.image('images/RF_pred.png',width=500)

        st.write("From the classification report we can see that, most of our dataset is in the range of 1. It predicts well based on the recall and precision.  \n"

                "### Precision — What percent of your predictions were correct?  \n"
                "Precision is the ability of a classifier not to label an instance positive that is actually negative. "
                "For each class, it is defined as the ratio of true positives to the sum of a true positive and false positive.  \n"
                "Precision:- Accuracy of positive predictions  \n"

                "In all 3 categories, the precision is high which is good  \n"

                "### Recall — What percent of the positive cases did you catch?  \n"
                "Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true"
                "positives to the sum of true positives and false negatives.  \n"
                "Recall:- Fraction of positives that were correctly identified  \n"

                "In the category of NextPurchaseDayRange =1 , we have the highest recall as most of our classification is under that bracket  \n"

                "### F1 score — What percent of positive predictions were correct?  \n"
                "The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. "
                "F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, "
                "the weighted average of F1 should be used to compare classifier models, not global accuracy.  \n"

                "We have a high F1 score,which is a good metric \n")

        st.write(df_pred_rf_model.tail(10))

        df_pred_purchase.loc[df_pred_purchase['customer_unique_id'] == '79444cb5bb16964eea4c5abb2d3aa023']

        df_user.loc[df_user['customer_unique_id'] == '79444cb5bb16964eea4c5abb2d3aa023']

        st.write(" From the above prediction,we have taken one customer id :  \n"

                "- the customer's NextPurchaseDayRange = 2 (which is customer will buy more than a month) but our model predicted the NextPurchaseDayRange = 1 "
                "(which is customer will buy less than a month)  \n"
                "- the customer is segmented as a low-value customer and we dont need to spend too much trying to re-acquire  \n"

                "The other customer's are predicted in the correct range which shows the customer will buy in less than a month at Olist which is NextPurchaseDayRange = 1")


   

main()

