# BatchServingMlModelContainer
A docker container to perform batch serving of a ML mode
## train.py uses 3 different classification models (Linear Discriminant Analysis (LDA), Neural Network (NN), Decision Tree Classifier (DT)) to train the model for training data ('train.csv')
## All 3 model joblib files will be saved under modelfiles folder.
## Model files will be used by test data ('test.csv') to predict scope and classification report in inference.py
## Docker image will be created using Dockerfile with required requirements as per 'requirements.txt'

## To test it, Pull from repository and test it.
## Happy coding and learning :)
