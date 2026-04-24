import pandas as pd
import os

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)

# Load datasets
customers = pd.read_csv("dataset/olist_customers_dataset.csv")
order_items = pd.read_csv("dataset/olist_order_items_dataset.csv")
order_payments = pd.read_csv("dataset/olist_order_payments_dataset.csv")
order_reviews = pd.read_csv("dataset/olist_order_reviews_dataset.csv")
orders = pd.read_csv("dataset/olist_orders_dataset.csv")
products = pd.read_csv("dataset/olist_products_dataset.csv")
sellers = pd.read_csv("dataset/olist_sellers_dataset.csv")
product_translation = pd.read_csv("dataset/product_category_name_translation.csv")

# Merge datasets
# Merge orders with customers
merged = orders.merge(customers, on="customer_id", how="left")

# Merge with order items
merged = merged.merge(order_items, on="order_id", how="left")

# Merge with payments
merged = merged.merge(order_payments, on="order_id", how="left")

# Merge with reviews
merged = merged.merge(order_reviews, on="order_id", how="left")

# Merge with products
merged = merged.merge(products, on="product_id", how="left")

# Merge with sellers
merged = merged.merge(sellers, on="seller_id", how="left")

# Merge with product category translations
merged = merged.merge(product_translation, on="product_category_name", how="left")

# Save the merged dataset
merged.to_csv("dataset/merged_dataset.csv", index=False)

# print("Datasets merged successfully into 'dataset/merged_dataset.csv'")