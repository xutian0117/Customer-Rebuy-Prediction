import joblib
import numpy as np

# Load the trained model
model = joblib.load('models/rebuy_model.pkl')

def predict_rebuy(customer_data):
    """
    Predict if a customer will make a repeat purchase.

    Parameters:
    customer_data (list): A list of customer attributes in the following order:
        - age
        - previous_purchases
        - discount_received
        - days_since_last_purchase
        - avg_purchase_value
        - gender
        - location
        - customer_service_interactions
        - personalized_recommendations

    Returns:
    int: 1 if the customer is likely to make a repeat purchase, 0 otherwise.
    """
    customer_data = np.array(customer_data).reshape(1, -1)
    return model.predict(customer_data)[0]

# Example usage of the prediction function
example_customer = [35, 3, 20, 30, 50, 'Male', 'NY', 1, 0]  # Example customer data
prediction = predict_rebuy(example_customer)
print(f"Will the customer make a repeat purchase? {'Yes' if prediction else 'No'}")
