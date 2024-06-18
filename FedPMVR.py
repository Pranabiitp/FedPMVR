#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)



def create_clients(data_dict):
    '''
    Return a dictionary with keys as client names and values as data and label lists.
    
    Args:
        data_dict: A dictionary where keys are client names, and values are tuples of data and labels.
                    For example, {'client_1': (data_1, labels_1), 'client_2': (data_2, labels_2), ...}
    
    Returns:
        A dictionary with keys as client names and values as tuples of data and label lists.
    '''
    return data_dict




def test_model(X_test, Y_test,  model, comm_round):
#     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
#     logits = model.predict(X_test)
    #print(logits)
    loss,accuracy=model.evaluate(X_test,Y_test)
#     loss = cce(Y_test, logits)
#     acc = accuracy_score( tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, accuracy, loss))
    return accuracy, loss


# In[5]:




import tensorflow as tf

def avg_weights(scaled_weight_list):
    '''Return the average of the listed scaled weights.'''
    num_clients = len(scaled_weight_list)
    
    if num_clients == 0:
        return None  # Handle the case where the list is empty
        
    avg_grad = list()
    
    # Get the sum of gradients across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0) / num_clients
        avg_grad.append(layer_mean)
        
    return avg_grad


import tensorflow as tf

# Load CIFAR-10 dataset
(_, _), (test, test_labels) = tf.keras.datasets.cifar10.load_data()

# Convert test labels to one-hot encoded format
num_classes = 10  # CIFAR-10 has 10 classes
test_labels_one_hot = tf.one_hot(test_labels, num_classes)

# Print the shape of the one-hot encoded labels
print("Shape of one-hot encoded test labels:", test_labels_one_hot.shape)
one_hot_labels=test_labels_one_hot
label=one_hot_labels
label=label.numpy()
# Remove the extra dimension from the one-hot encoded labels
label = label.squeeze(axis=1)

# Print the updated shape of the labels
print("Updated shape of labels:", label.shape)


# In[8]:


#import training data before as mentioned in readme file  
for i in range(1, 11):
    globals()[f"train{i}"] = globals()[f"train{i}"] / 255



test=test/255


# In[9]:


client_data1 = {
#     'client0': (test, label),
    'client1': (test, label),
    'client2': (test, label),
    'client3': (test, label),
    'client4': (test, label),
    'client5': (test, label),
    'client6': (test, label),
    'client7': (test, label),
    'client8': (test, label),
    'client9': (test, label),
    'client10': (test, label)


#     'client6': (test, label)
    
}
#create clients
test_batched = create_clients(client_data1)
client_data2 = {
    'client1': (train1, label1),
    'client2': (train2, label2),
    'client3': (train3, label3),
    'client4': (train4, label4),
    'client5': (train5, label5),
    'client6': (train6, label6),
    'client7': (train7, label7),
    'client8': (train8, label8),
    'client9': (train9, label9),
    'client10': (train10, label10)

    
}
#create clients
clients_batched = create_clients(client_data2)


client_names = list(clients_batched.keys())




import numpy as np
import matplotlib.pyplot as plt

# Number of global epochs
comms_round = 200
acc3 = []
loss3 = []
train_acc_clients = [[], [], [], []]  # List of lists for training accuracy for each client
val_acc_clients = [[], [], [], []]    # List of lists for validation accuracy for each client
best_acc = 0
best_weights = None

# Initialize momentum terms for each client
momentum_terms = {client: [np.zeros_like(w) for w in global_model.get_weights()] for client in clients_batched.keys()}

for comm_round in range(comms_round):

    # Get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # Initial list to collect local model weights after scaling
    local_weight_list = []

    # Randomize client data - using keys
    client_names = list(clients_batched.keys())
    # random.shuffle(client_names)

    for client in client_names:

        local_model = create_cnn_model()

        # Set local model weight to the weight of the global model
        local_model.set_weights(global_weights)

        # Fit local model with client's data
        history = local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),
            validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
            epochs=1,
            batch_size=16,
            verbose=2
        )

        # Store the training accuracy and validation accuracy for this client in this communication round
        # train_acc_clients[client_names.index(client)].append(history.history['accuracy'][0])
        # val_acc_clients[client_names.index(client)].append(history.history['val_accuracy'][0])

        # Get the model weights and calculate the gradient
        new_weights = local_model.get_weights()
        gradients = [(new - old) for new, old in zip(new_weights, global_weights)]

        # Update momentum terms for the last two layers
        alpha = 0.1  # Learning rate for momentum update
        for i in range(-2, 0):  # Only update momentum for the last two layers
            momentum_terms[client][i] = alpha * gradients[i] + momentum_terms[client][i]

        # Update weights with momentum correction for the last two layers
        corrected_weights = [(new_weights[i] - momentum_terms[client][i]) if i in range(-2, 0) else new_weights[i]
                             for i in range(len(new_weights))]
        
        local_weight_list.append(corrected_weights)

        # Clear the session to free memory after each communication round
        # K.clear_session()

    # Calculate the average weights across all clients for each layer
    average_weights = avg_weights(local_weight_list)

    # Update the global model with the average weights
    global_model.set_weights(average_weights)

    # Test the global model and print out metrics after each communications round
    global_acc, global_loss = test_model(test, label, global_model, comm_round)
    acc3.append(global_acc)
    loss3.append(global_loss)

    plt.plot(acc3, label='Accuracy')
    plt.plot(loss3, label='Loss')
    plt.legend()
    plt.show()

    if global_acc > best_acc:
        best_acc = global_acc
        best_weights = global_model.get_weights()
        # global_model.save("fedprox_full_isic_new1.h5")

global_model.set_weights(best_weights)







