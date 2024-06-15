## FedPMVR: Addressing Data Heterogeneity in Federated Learning through Partial Momentum Variance Reduction
This repository contains the code for the paper FedPMVR: Addressing Data Heterogeneity in Federated Learning through Partial Momentum Variance Reduction.

# Dependencies
- Tensorflow = 2.10.0
- scikit-learn = 1.3.2

# Data Preparing
To divide the dataset into the aprequired no. of clients, run Data Prepration.py and choose the required dataset (CIFAR10, MNIST or FMNIST) and then change the degree of heterogenity (beta) as required. you will get the desired distribution for each client.

# Model Structure
To choose the appropriate  model, run Models.py and choose the required model for each of the dataset.

# Run FedPMVR
After done with above process, you can run the FedPMVR, our proposed method.

# Evaluation
After federated training, run Evaluation.py to acess the evaluation metrics such as accuracy, precision, recall etc.

