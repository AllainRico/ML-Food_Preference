#The system aims to determine the gender of a person based on his/her food preference
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
# Load the CSV
dataset = pd.read_csv('FoodPreference.csv')
# Graph
# Convert strings to numeric
dataset.Gender = dataset.Gender.replace(to_replace=['Female', 'Male'], value=[0, 1])
dataset.Food = dataset.Food.replace(to_replace=["Traditional food", "Western Food"], value=[0, 1])
dataset.Juice = dataset.Juice.replace(to_replace=['Fresh Juice', 'Carbonated drinks'], value=[0, 1])
dataset.Dessert = dataset.Dessert.replace(to_replace=['Yes', 'No'], value=[0, 1])

# Create the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(dataset[['Age', "Food", "Juice", "Dessert"]], dataset.Gender)
# Save the model
with open('logistic.pk', 'wb') as f:
	pickle.dump(model, f)

# Test the model
t_Age = 30
t_Food = "Traditional food"
t_Juice = "Fruit Juice"
t_Dessert = "No"

t_Food = 1 if t_Food == 'Western Food' else 0
t_Juice = 1 if t_Juice == 'Carbonated drinks' else 0
t_Dessert = 1 if t_Dessert == 'No' else 0

output = model.predict_proba([[t_Age, t_Food, t_Juice, t_Dessert]])
print("Female", "{:.4f}".format(output[0][0]))
print("Male", "{:.4f}".format(output[0][1]))