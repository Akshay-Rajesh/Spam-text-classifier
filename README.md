# SMS Spam Classifier Project

## **Project Description**
This project aims to build a binary classification model using Naive Bayes to identify SMS messages as either *spam* or *ham* (not spam). The model will be trained and evaluated using the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

## **Project Structure**
The project is organized into the following files:
- **`main.py`**: Orchestrates data loading, preprocessing, model training, and evaluation.
- **`data_loader.py`**: Handles dataset loading and preprocessing, including tokenization and vectorization.
- **`model.py`**: Defines and trains the Naive Bayes model.
- **`evaluate.py`**: Evaluates the model's performance using metrics such as precision, recall, and F1-score.
- **`requirements.txt`**: Lists all dependencies required for the project.

## **Technologies Used**
- Python
- Scikit-learn
- Pandas
- NumPy
- Dockers

## **How to Run**
1. Clone this repository to your local machine.
2. Install the dependencies using the following command:
   ```bash
   pip install -r requirements.txt

## Best performance so far

Model:NB 
Preprocessing : True
accuracy: 0.9757847533632287
f1_score: 0.9120521172638436
precision: 0.8860759493670886
recall: 0.9395973154362416

## Model in Dagshub

https://dagshub.com/Akshay-Rajesh/Spam-text-classifier/experiments#/experiment/m_d519a7cfcb7548b6b01e909175df5423

## **Wiki-page for Weekly Minutes
- [https://github.com/pffaundez/Mini-Projects.wiki.git](https://github.com/pffaundez/Mini-Projects/wiki)
