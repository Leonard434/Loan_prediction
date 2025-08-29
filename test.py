import pickle

# Load the saved model
with open("best_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Example prediction
sample_input = [[1, 1, 2.0, 0, 0, 5000, 1000.0, 150.0, 360.0, 1.0, 1]]  # dummy input
prediction = loaded_model.predict(sample_input)

print("Prediction:", prediction)
