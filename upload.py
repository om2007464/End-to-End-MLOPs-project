import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
# Replace this with your local MongoDB URI or MongoDB Atlas URI
client = MongoClient("mongodb+srv://Spiderman:Spiderman@cluster0.ctt8i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Use MongoDB Atlas URI if applicable
db = client["Spiderman"]  # Database name
collection = db["raw_data"]  # Collection name

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv("/Users/om/Projects/git p/Gemstone-Price-Prediction/artifacts/raw.csv")  # Update with your actual CSV path

# Convert the DataFrame to a dictionary format
data_dict = data.to_dict(orient='records')

# Insert the data into MongoDB
collection.insert_many(data_dict)

print("Data uploaded to MongoDB successfully!")
