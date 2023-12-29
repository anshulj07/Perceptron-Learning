import numpy as np

class PerceptronHW1:
    def __init__(self, data_path="train-a1-449.txt", dims=1024):
        self.setup_model(data_path, dims)

    def setup_model(self, data_path, dims):
        data = np.loadtxt(data_path, skiprows=1, dtype='str')
        self.inputs = data[:, :dims].astype(float)
        self.outputs = data[:, dims:]
        self.weights = np.zeros(dims)
        self.lr = 0.5
        self.margin = 0

    def train_model(self):
        for _ in range(100):
            self.process_inputs()

    def process_inputs(self):
        for input_vector, output_label in zip(self.inputs, self.outputs):
            prediction = self.predict_output(input_vector)
            self.update_model_weights(input_vector, output_label, prediction)

    def update_model_weights(self, input_vector, output_label, prediction):
        correct_output = 1 if output_label[0] == 'Y' else -1
        adjustment = self.lr * (prediction - correct_output) * input_vector
        self.weights += adjustment
        self.margin = self.calculate_margin(input_vector)

    def predict_output(self, input_vector):
        return 1 if np.dot(input_vector, self.weights) > 0 else -1

    def calculate_margin(self, input_vector):
        return np.linalg.norm(self.weights - input_vector)

    def normalize_weights(self):
        self.weights /= np.linalg.norm(self.weights)

if __name__ == '__main__':
    model = PerceptronHW1()
    model.train_model()
    model.normalize_weights()
    np.savetxt("model_weights.txt", model.weights)
    with open("model_margin.txt", "w") as file:
        file.write(str(model.margin))