import pandas as pd
import numpy as np

def feature_engineering(data):
    """
    Perform feature engineering on the customer purchase history data.

    Parameters:
    data (pd.DataFrame): The original customer data.

    Returns:
    pd.DataFrame: The transformed customer data with new features.
    """

    # Feature 1: Personalized Product Recommendations
    data['personalized_recommendations'] = data['previous_purchases'].apply(lambda x: 1 if x > 5 else 0)

    # Feature 2: Targeted Discounts
    data['received_discount'] = data['discount_received'].apply(lambda x: 1 if x > 0 else 0)

    # Feature 3: Customer Service Interactions
    data['customer_service_interactions'] = data['customer_service_calls'].apply(lambda x: 1 if x > 2 else 0)

    # Feature 4: Time Since Last Purchase
    data['days_since_last_purchase'] = (pd.to_datetime('today') - pd.to_datetime(data['last_purchase_date'])).dt.days

    # Feature 5: Average Purchase Value
    data['avg_purchase_value'] = data['total_spent'] / data['previous_purchases']

    # Feature 6: Price Sensitivity
    # Hypothesis: The price of products purchased may influence the likelihood of repeat purchases.
    data['price_sensitivity'] = data['price'].apply(lambda x: 'high' if x > data['price'].mean() else 'low')

    # Feature 7: Product Line Popularity
    # Hypothesis: Customers purchasing from certain product lines may be more likely to return.
    product_line_purchase_counts = data.groupby('product_line')['product_line'].transform('count')
    data['product_line_popularity'] = product_line_purchase_counts / len(data)

    # Feature 8: Quantity Purchased
    # Hypothesis: Customers who purchase higher quantities may be more inclined to repurchase.
    data['high_quantity'] = data['quantity'].apply(lambda x: 1 if x > data['quantity'].median() else 0)

    # Feature 9: Specific Product Indicator
    # Hypothesis: Certain products may drive higher repeat purchases.
    top_products = data['product'].value_counts().index[:5]
    data['top_product_indicator'] = data['product'].apply(lambda x: 1 if x in top_products else 0)

    return data

# Example usage:
# data = pd.read_csv('customer_purchase_history.csv')
# transformed_data = feature_engineering(data)
# print(transformed_data.head())
