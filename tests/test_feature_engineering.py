import pandas as pd
from src.feature_engineering import feature_engineering

def test_feature_engineering():
    # Example input data
    data = pd.DataFrame({
        'previous_purchases': [1, 10, 4],
        'discount_received': [0, 50, 10],
        'customer_service_calls': [0, 3, 1],
        'last_purchase_date': ['2023-08-01', '2023-07-01', '2023-08-10'],
        'total_spent': [100, 1000, 300],
        'price': [20, 200, 60],
        'product_line': ['A', 'B', 'C'],
        'quantity': [1, 5, 2],
        'product': ['P1', 'P2', 'P3']
    })

    # Apply feature engineering
    transformed_data = feature_engineering(data)

    # Assertions to check if the features are correctly created
    assert 'personalized_recommendations' in transformed_data.columns
    assert 'received_discount' in transformed_data.columns
    assert 'customer_service_interactions' in transformed_data.columns
    assert 'days_since_last_purchase' in transformed_data.columns
    assert 'avg_purchase_value' in transformed_data.columns
    assert 'price_sensitivity' in transformed_data.columns
    assert 'product_line_popularity' in transformed_data.columns
    assert 'high_quantity' in transformed_data.columns
    assert 'top_product_indicator' in transformed_data.columns

if __name__ == "__main__":
    test_feature_engineering()
    print("All tests passed!")
