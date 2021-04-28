# CIS4930 Natural Language Processing with Python Project - Fake News Classification

## Abinai Pasunuri, Ammar Syed

### Code Base Structure

The following is the code base of the project. It consists of four main `.py` files:

- `main.py`: It is the main program, which loads in the preprocessed data, initializes the two models, trains the models, and generates the different evaluation metrics and plots of the models.
- `data.py`: It contains various functions to scrape, load, and preprocess the news articles corpus. Program starts with scraping all news articles contents from provided urls, with the Python Newspaper library. It then combines the local CSV dataset with the scraped dataset, and lastly preprocesses the combined data, with Pandas, Numpy, and NLTK libraries.
- `models.py`: It contains classes of the different classifier models. It includes a class for the Support Vector Machines model, where the model is initialized with the scikit-learn library. It also contains a class for the Convolutional-Recurrent Neural Network, where the architecture is constructed through Keras and Tensorflow libraries.
- `visualizations.py`: It contains various functions that takes in various data, to construct Confusion Matrices and ROC Curves for the classifers and a Class Distribution Bar Chart and Word Clouds for the combined total Preprocessed Dataset.

The code base also contains three main directories.

- `Data`: It contains all the CSV files of the news articles and their corresponding labels.
- `Figures`: It contains all the PNG files of the different figures, such as Confusion Matrices, ROC Curves.
- `Models`: It contains the Pickle file of the trained SVM model.

### Installation of Libraries

The `requirements.txt` contains a list of all Python libraries and cooresponding versions of the libraries the code runs on. In order to install these libraries run the command `pip install -r requirements.txt` in a Python environment to get all the necessary libraries installed and to run the code.

### Running Code

In order to run the code, the main file must be run with the command `python main.py "model_name"`, where `model_name` is a string command line argument that represents the type of model the program will run. Input `"svm"` in place of `"model_name"` in order to run the program for the Support Vector Machines model and input `"nn"` in place of `"model_name"` in order to run the program for the Neural Network model. To run other supported models, input `"logreg"` for a Logistic Regression Model, `"nb"` for a Naive Bayes Model, and `"dt"` for a Decision Tree model. If any other string is inputted for the command line argument, then the program will simply display the output `Invalid Model Type` to the console and terminate.

### Code Output

Once the code is run with the above command, the program will start with getting the preprocessed news articles corpora. If this does not already exist in the `Data` directory it will generate through running the preprocess function and scraping the data. It will then vectorize the data and split it into training and testing sets.

It will then train the corresponding model specified in the command line argument and following the training it will display the metrics of the model on the testing data to the console, including Accuracy, AUC, Precision, and Recall. The program will also run the different visualization functions and generate PNG files in the `Figures` directory of the different metrics of the dataset and corresponding model. If a saved model file already exists in the `Models` directory, then the program will skip training the model and will just load the model and evaluate it.
