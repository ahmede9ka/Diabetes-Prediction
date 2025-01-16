import unittest
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

class TestDiabetesModel(unittest.TestCase):
    def setUp(self):
        with open('../diabetes.pkl', 'rb') as file:
            self.model = pickle.load(file)
        
        # Recreate scaler used during training
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([3.8, 120.9, 69.1, 20.5, 80.3, 31.2, 0.47, 33.2])  # Replace with your actual scaler stats
        self.scaler.scale_ = np.array([3.5, 30.2, 19.3, 15.9, 115.2, 7.9, 0.33, 11.8])

    def test_prediction_no_diabetes(self):
        input_data = np.array([[1, 85, 70, 20, 85, 22.5, 0.2, 25]])
        input_data_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_data_scaled)
        print(f"Test Input: {input_data_scaled}")
        print(f"Prediction: {prediction}")
        self.assertEqual(prediction[0], 0)  # Update expected value based on manual app testing

    def test_prediction_diabetes(self):
        input_data = np.array([[6, 190, 88, 33, 130, 30.0, 0.6, 50]])
        input_data_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_data_scaled)
        print(f"Test Input: {input_data_scaled}")
        print(f"Prediction: {prediction}")
        self.assertEqual(prediction[0], 1)

if __name__ == '__main__':
    unittest.main()
