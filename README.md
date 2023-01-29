# WIDS-Image-caption-generation-using-deep-learning

## Week 1- 
- This project is focused on using machine learning algorithms to predict individual product failures.
- The dataset provided includes product attributes and measurement values obtained from lab testing methods, and contains both numerical and categorical features. The target column is "failure". 
- The project will use several algorithms such as XGBoost, KNN Classifier, Decision Tree, Naive Bayes, Logistic Regression and Neural Networks. 
- The evaluation metric for the model will be the Area under the ROC curve between the predicted probability and the observed target. 
- Data pre-processing techniques such as basic EDA, feature selection, elimination, and transformation will also be applied. 
- The final submission will be a Jupyter notebook file that clearly separates each step of the prediction pipeline and provides explanations for each step taken in markdown.


## Week 2-

- Trained a basic neural network for recognizing handwritten digits from the MNIST dataset
- Learned key concepts of backpropagation, activation functions, preprocessing, and hyperparameter tuning
- Built and trained a neural network from scratch
- Evaluated the performance using metrics like accuracy, precision, recall and F1-score
- Implemented techniques like k-fold cross-validation for model evaluation
- Achieved impressive accuracy of 98%

##  Week 3-
- In the third week, we delved deeper into convolutional neural networks (CNNs), using them to classify images in the CIFAR-10 dataset.
- We gained hands-on experience implementing CNNs and exploring their intricacies.
- We learned about the different types of layers used in CNNs such as convolutional layers, pooling layers, and padding and their function in the architecture.
- Additionally, we experimented with different architectures and hyperparameters to understand how they affect the performance of the model.
- We also learned about popular CNN architectures such as LeNet, AlexNet, VGG and ResNet and the role they play in the computer vision field.

## Week 4:

- The fourth week presented two challenges: predicting oil prices for the next 30 days using historical data, and performing sentiment analysis on stock market statements or headlines.
- For the first challenge, we studied Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models. We learned about the working principle of RNNs, and how they are used to process sequential data. We also learned about the LSTM architecture and how it addresses the problem of vanishing gradients that occurs in traditional RNNs.
- We trained an LSTM model to predict oil prices using historical data and evaluated its performance using metrics such as mean absolute error and mean squared error.
- For the second challenge, we applied basic natural language processing (NLP) techniques like removing stopwords, tokenization and lemmatization to predict sentiments.
- We learned about the importance of text preprocessing and feature extraction in NLP tasks, and how to use techniques like bag-of-words and term frequency-inverse document frequency (TF-IDF) to represent text data.

## Week 5:

- The final week was the culmination of all previous weeks, as we were tasked with generating image captions for the Flickr8k dataset.
- We utilized a pre-trained neural network, VGG 19, for feature extraction from images. This allowed us to extract a feature vector that represents the image and reduces the dimensionality of the input data.
- We then passed these feature vectors through an LSTM network which we trained to generate captions that describe the images.
- We learned about the concept of transfer learning and how to use pre-trained models to improve performance. This is an important technique in deep learning where a model that has been trained on one task can be used as a starting point for a related task, allowing us to train models more quickly and with better performance.
- We also learned about the architecture of LSTM and how it is used to generate captions. LSTMs are powerful models that are able to process sequential data and retain information over long periods of time. This makes them well-suited for tasks like caption generation where the output is a sequence of words.
- We evaluated the performance of the model using BLEU. These metrics are commonly used to evaluate the quality of generated captions.
- Finally, we fine-tuned the model to improve performance by experimenting with different architectures, hyperparameters, and techniques like beam search. We also learned about other techniques like attention mechanism which can improve the performance of the caption generation model
- Overall, this week gave us a comprehensive understanding of image caption generation and how to use pre-trained models, LSTMs and other techniques to improve performance.
