#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)


# In[2]:


import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
# !pip install fl_implementation_utils

# from fl_implementation_utils import *
import tensorflow as tf

def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Conv2D(6, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(),
                  loss=CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    return model
global_model = create_cnn_model()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:




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


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:



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


# In[18]:





# In[6]:


import numpy as np
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load CIFAR-10 dataset
cifar10_data = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10_data.load_data()

# Shuffle the dataset
shuffle_indices = np.random.permutation(len(train_images))
train_images_shuffled = train_images[shuffle_indices]
train_labels_shuffled = train_labels[shuffle_indices]

# Number of clients
num_clients = 10

# Number of classes in CIFAR-10
num_classes = 10

# Simulate heterogeneous partition using Dirichlet distribution
# Here, we'll assume equal proportions for simplicity
proportions = np.random.dirichlet(np.ones(num_clients) * 0.1, size=num_classes)

# Allocate data to clients
client_data_indices = [[] for _ in range(num_clients)]
for class_label in range(num_classes):
    num_samples_per_class = np.sum(train_labels_shuffled == class_label)
    for client_idx in range(num_clients):
        num_samples_for_client = int(proportions[class_label, client_idx] * num_samples_per_class)
        # Randomly select data indices for the client
        selected_indices = np.random.choice(
            np.where(train_labels_shuffled[:,0] == class_label)[0],
            size=num_samples_for_client,
            replace=False
        )
        client_data_indices[client_idx].extend(selected_indices)

# Now client_data_indices contains the indices of data samples allocated to each client
# Initialize lists to store data and labels for each client
client_train_data = []
client_train_labels = []

# Extract data and corresponding one-hot labels for each client
for client_indices in client_data_indices:
    client_data_samples = train_images_shuffled[client_indices]
    client_data_labels = train_labels_shuffled[client_indices]
    
    # Convert labels to one-hot encoding
    one_hot_labels = tf.keras.utils.to_categorical(client_data_labels, num_classes)
    
    # Append data and labels for this client to the list
    client_train_data.append(client_data_samples)
    client_train_labels.append(one_hot_labels)

# Naming the variables as train1, label1, train2, label2, and so on for all 10 clients
for i in range(len(client_train_data)):
    globals()[f"train{i+1}"] = client_train_data[i]
    globals()[f"label{i+1}"] = client_train_labels[i]


# In[1]:


# train4.shape


# In[2]:


# import matplotlib.pyplot as plt

# # Define very dark colors for each client
# client_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# plt.figure(figsize=(12, 8))
# for client_idx, client_samples in enumerate(client_data):
#     class_counts = {class_label: 0 for class_label in range(num_classes)}
#     for sample_idx in client_samples:
#         class_label = train_labels[sample_idx][0]  # Extracting the label from the loaded labels array
#         class_counts[class_label] += 1
#     plt.bar([str(class_label) for class_label in range(num_classes)], class_counts.values(), alpha=0.5, color=client_colors[client_idx], label=f'Client {client_idx + 1}')

# plt.xlabel('Class Label')
# plt.ylabel('Number of Samples')
# plt.title('Data Distribution for Each Client')
# plt.legend()
# plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Create pixel vals array with 255 values to match histogram bins
pixel_vals = np.arange(0, 256)

# Initialize the figure and axis
fig, ax = plt.subplots()

# Iterate over all clients
for i in range(1, 11):
    # Extract training data for the current client
    train = locals()[f'train{i}']
    
    # Calculate density
    density = gaussian_kde(train.reshape(-1))
    
    # Evaluate density at each pixel value
    y = density.evaluate(pixel_vals)
    
    # Plot the density curve for the current client
    ax.plot(pixel_vals, y, label=f'Client {i}')

# Add legend and labels
ax.legend()
ax.set_xlabel('Grayscale value')
ax.set_ylabel('Pixels')

# Show the plot
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Create pixel vals array with 255 values to match histogram bins
pixel_vals = np.arange(0, 256)

# Initialize the figure and axis
fig, ax = plt.subplots()

# Define color scheme for clients
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Iterate over all clients
for i in range(1, 11):
    # Extract training data for the current client
    train = locals()[f'train{i}']
    
    # Calculate density
    density = gaussian_kde(train.reshape(-1))
    
    # Evaluate density at each pixel value
    y = density.evaluate(pixel_vals)
    
    # Plot the density curve for the current client
    ax.plot(pixel_vals, y, label=f'Client {i}', color=colors[i-1], alpha=0.7)

# Add legend and labels
ax.legend(loc='upper right')
ax.set_title('Pixel Intensity Distribution Across Clients')
ax.set_xlabel('Pixel Intensity')
ax.set_ylabel('Density')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[3]:


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


# In[4]:


# label.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# initialize global model
# print(data_list.shape,labels)

        


# In[ ]:


global_model.summary()


# In[5]:


# len(clients_batched[client][1])
# global_model.get_weights()
client_names = list(clients_batched.keys())
client_names


# In[ ]:



label1.shape


# In[ ]:


np.min(train4)


# In[6]:


comms_round = 200  # Number of global epochs
acc3 = []
loss3=[]
train_acc_clients = [[], [], [],[]]  # List of lists for training accuracy for each client
val_acc_clients = [[], [], [],[]]    # List of lists for validation accuracy for each client
best_acc = 0
best_weights = None


for comm_round in range(comms_round):

    # Get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # Initial list to collect local model weights after scaling
    local_weight_list = []

#     Randomize client data - using keys
    client_names = list(clients_batched.keys())
#     random.shuffle(client_names)

    for i, client in enumerate(client_names):

        local_model = create_cnn_model()

#         local_model.compile(
#             loss='categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy']
#         )

        # Set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
#         num_local_epochs = client_epochs[client]
#         for _ in range(num_local_epochs):
#                 history = local_model.fit(
#                     np.array(clients_batched[client][0]),
#                     np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),np.array(clients_batched[client][1])),
#                     epochs=1,
#                     verbose=2
#                 )

        # Fit local model with client's data
        if client == 'client1':
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=1,
            batch_size=16,
            verbose=2
        )
        elif client == 'client2':
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=1,
            batch_size=16,
            verbose=2
        )
            
        elif client == 'client3':
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=1,
            batch_size=16,
            verbose=2
        ) 
        else:
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=1,
            batch_size=16,
            verbose=2
        ) 
            
            
        
        # Store the training accuracy and validation accuracy for this client in this communication round
#         train_acc_clients[i].append(history.history['accuracy'][0])
#         val_acc_clients[i].append(history.history['val_accuracy'][0])

        # Get the scaled model weights and add to the list
        weights = local_model.get_weights()
        local_weight_list.append(weights)

        # Clear the session to free memory after each communication round
        K.clear_session()

    # Calculate the average weights across all clients for each layer
    average_weights = avg_weights(local_weight_list)

    # Update the global model with the average weights
    global_model.set_weights(average_weights)

    # Test the global model and print out metrics after each communications round
#     for (X_test, Y_test) in test_batched:
    global_acc, global_loss = test_model(test, label, global_model, comm_round)
    acc3.append(global_acc)
    loss3.append(global_loss)
    import matplotlib.pyplot as plt
    plt.plot(acc3)
    plt.plot(loss3)
    if global_acc > best_acc:
        best_acc = global_acc
        best_weights = global_model.get_weights()
#          global_model.save("fedprox_full_isic_new1.h5")   
        
global_model.set_weights(best_weights)        


# In[ ]:





# In[7]:


# # import matplotlib.pyplot as plt
# plt.plot(acc3,color='red')
# plt.title("FL vs communication rounds")
# plt.grid(visible=True)
# plt.legend(loc='right')
# # plt.ylim(.6,.9)
# plt.xlabel("FL rounds/Data sharing epochs")
# plt.ylabel("Test accuracy")


# In[8]:


# global_model.evaluate(test,label)


# In[9]:


from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, f1_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

# Assuming you have predictions and true labels
y_true = label # Replace with your true labels
y_pred = global_model.predict(test)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate Accuracy
acc = accuracy_score(y_true, y_pred)

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(y_true, y_pred)

# Calculate Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_true, y_pred)

# Calculate Balanced Accuracy
bacc = balanced_accuracy_score(y_true, y_pred)

# Calculate F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multiclass

# Calculate Precision for multiclass
precision = precision_score(y_true, y_pred, average='weighted')

# Calculate Recall for multiclass
recall = recall_score(y_true, y_pred, average='weighted')



# # Calculate AUC (Area Under the Curve)
# # roc_auc = roc_auc_score(y_true, y_pred, average='weighted')

# # Create a confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)

# # Calculate Geometric Mean from the confusion matrix
# # tn, fp, fn, tp, = conf_matrix.ravel()
# # g_mean = (tp / (tp + fn)) * (tn / (tn + fp))**0.5

# # Print or use these metrics as needed
print("Accuracy:", acc)
print("Cohen's Kappa:", kappa)
print("Matthews Correlation Coefficient:", mcc)
print("Balanced Accuracy:", bacc)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
# print("AUC (Area Under the Curve):", roc_auc)
# print("Geometric Mean:", g_mean)
from sklearn.metrics import roc_auc_score

# Assuming you have true labels and predicted probabilities for each class
y_true = label  # Replace with your true labels
y_prob = global_model.predict(test)  # Replace with your predicted probabilities

# # Calculate AUC for multiclass classification
auc = roc_auc_score(y_true, y_prob, average='weighted')

# # Print or use the AUC value as needed
print("AUC (Area Under the Curve):", auc)
from sklearn.metrics import confusion_matrix
import numpy as np

# Assuming you have true labels and predicted labels for multiclass classification
y_true = one_hot_labels  # Replace with your true labels
y_pred = global_model.predict(test)  # Replace with your predicted labels

# Convert true and predicted labels to class labels (not one-hot encoded)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate G-Mean for each class
class_g_means = []
for class_label in range(5):  # Replace num_classes with the number of classes
    # Create a binary confusion matrix for the current class
    true_class = (y_true == class_label)
    pred_class = (y_pred == class_label)
    tn, fp, fn, tp = confusion_matrix(true_class, pred_class).ravel()

    # Calculate Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate G-Mean for the current class
    g_mean = np.sqrt(sensitivity * specificity)

    class_g_means.append(g_mean)

# Calculate the overall G-Mean (geometric mean of class G-Means)
overall_g_mean = np.prod(class_g_means) ** (1 / len(class_g_means))

# Print or use the overall G-Mean as needed
print("Overall G-Mean:", overall_g_mean)


# In[ ]:


plt.plot(range(1,len(acc3)+1),train_acc_clients[0],label='train')
plt.plot(range(1,len(acc3)+1),val_acc_clients[0],color='red',label='val')
# plt.ylim(0.95,.999)


# In[ ]:


plt.plot(range(1,len(acc3)+1),train_acc_clients[1])
plt.plot(range(1,len(acc3)+1),val_acc_clients[1],color='red')
plt.ylim(0.90,.999)


# In[ ]:


plt.plot(range(1,len(acc3)+1),train_acc_clients[0])
plt.plot(range(1,len(acc3)+1),val_acc_clients[0],color='red')
plt.ylim(0.5,.999)


# In[14]:


acccc=np.array(acc3)


# In[15]:


np.save("acc_fedavg_alpha=0.1",acccc)


# In[16]:


losss=np.array(loss3)
np.save("loss_fedavg_alpha=0.1",losss)


# In[17]:


global_model.save("fedavg(alpha=0.1).h5")


# In[ ]:


global_model.evaluate(test1,one_hot_labels1)


# In[ ]:


# !pip list


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




