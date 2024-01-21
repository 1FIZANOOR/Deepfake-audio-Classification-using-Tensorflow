# Deepfake-audio-Classification-using-Tensorflow
This Python script uses TensorFlow and Keras to build and train a SincNet model for speaker classification. 
The SincNet model is a type of neural network that is specifically designed for processing raw audio signals. 
The script begins by importing necessary libraries and defining a function `create_sincnet_model()` to construct the SincNet model. 
The model consists of several layers including a SincConv1D layer, several Conv1D layers with batch normalization, max pooling, and leaky ReLU activation functions. 
After flattening the output, dense layers are added for further processing. The final layer uses a sigmoid activation function for binary classification.

Next, the script defines a function `load_audio_file()` to load and preprocess audio files. 
It then specifies directories for real and fake audio files, loads the audio files, and reshapes them into the required shape for the model.
Labels are also assigned to the audio files based on whether they are real or fake. 
The data is split into training and testing sets using `train_test_split()`.

The script then creates a TensorFlow Dataset from the training data and shuffles and batches it for efficient training. 
The SincNet model is compiled with Adam optimizer and binary cross entropy loss function. 
The model is then fitted to the training data for 20 epochs, with the test data used for validation. 
The trained model is saved as "my_model.h5", and finally, the model's performance is evaluated on the test set.
