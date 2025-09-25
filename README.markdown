# News Text Preprocessing and Classification Project

## Overview

This project focuses on preprocessing and analyzing a collection of news articles from the BBC dataset (`bbc-text.csv`). The pipeline includes loading and exploring the data, preprocessing the text, extracting features using TF-IDF, training a RandomForestClassifier to classify news articles into categories, and evaluating the model's performance. Visualizations such as word clouds, frequency plots, and confusion matrices are generated to aid analysis. The final predictions are saved, and the project files are zipped for submission.

## Objectives

- Load and explore the BBC news dataset.
- Preprocess text data by converting to lowercase, removing punctuation and digits, tokenizing, removing stopwords, and applying lemmatization.
- Visualize data distributions and word frequencies using count plots and word clouds.
- Extract features using TF-IDF vectorization.
- Train a RandomForestClassifier to classify news articles by category.
- Evaluate the model using classification metrics and a confusion matrix.
- Save predictions and zip project files for submission.

## Project Structure

- `Project.ipynb`: The main Jupyter Notebook containing the code for data loading, preprocessing, visualization, model training, and evaluation.
- `predictions.csv`: Output file containing the test set indices, preprocessed text, and predicted categories.
- `submission.zip`: Zipped file containing `Project.ipynb` and `predictions.csv` for submission.
- `../Data/bbc-text.csv`: Input dataset (not included in the repository; must be provided).

## Requirements

To run this project, you need the following Python libraries:

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `wordcloud`

You can install the required libraries using pip:

```bash
pip install pandas numpy nltk scikit-learn seaborn matplotlib wordcloud
```

Additionally, download the required NLTK resources within the notebook:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Setup Instructions

1. **Clone or Download the Project**:
   - Clone this repository or download the project files.
   - Ensure the `bbc-text.csv` dataset is placed in the `../Data/` directory relative to the notebook.

2. **Set Up the Environment**:
   - Create a Python environment (Python 3.6 or higher recommended).
   - Install the required libraries listed above.

3. **Run the Notebook**:
   - Open `Project.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to load data, preprocess text, train the model, and generate outputs.

4. **Verify Outputs**:
   - The notebook generates `predictions.csv` and `submission.zip` in the working directory.

## Usage

1. **Run the Notebook**:
   - Open `Project.ipynb` and run all cells. The notebook will:
     - Load and display the dataset.
     - Visualize the distribution of news categories.
     - Preprocess the text and generate word clouds for each category.
     - Analyze word frequencies and TF-IDF features.
     - Train and evaluate a RandomForestClassifier.
     - Save predictions to `predictions.csv` and zip the project files into `submission.zip`.

2. **Inspect Outputs**:
   - Check `predictions.csv` for the model's predictions on the test set.
   - Extract `submission.zip` to verify the included files (`Project.ipynb`, `predictions.csv`).

3. **Submit Results**:
   - Use `submission.zip` for submission, as it contains the required files.

## Key Features

- **Text Preprocessing**: Converts text to lowercase, removes punctuation and digits, tokenizes, removes stopwords, and applies lemmatization.
- **Visualization**: Includes category distribution plots, word clouds, word frequency bar charts, and a confusion matrix.
- **Feature Extraction**: Uses TF-IDF to vectorize text for machine learning.
- **Model Training**: Employs a RandomForestClassifier for news category classification.
- **Evaluation**: Provides detailed metrics (precision, recall, F1-score) and a confusion matrix to assess model performance.

## Notes

- The dataset (`bbc-text.csv`) must be available in the `../Data/` directory for the notebook to run successfully.
- There is a minor bug in the preprocessing function where `text.lower()` is called without assigning the result, so text is not converted to lowercase. This does not affect the overall pipeline but may impact results slightly.
- The project uses lemmatization instead of stemming, despite the documentation mentioning both.
- The RandomForestClassifier is configured with 100 trees and a fixed random state for reproducibility.

## Output Files

- **`predictions.csv`**: Contains the test set indices, preprocessed text, and predicted categories.
- **`submission.zip`**: Contains `Project.ipynb` and `predictions.csv` for submission.

## License

This project is for educational purposes and uses the BBC news dataset. Ensure you have the right to use the dataset for your purposes.

## Contact

For any issues or questions, please contact the project maintainer or refer to the documentation within `Project.ipynb`.