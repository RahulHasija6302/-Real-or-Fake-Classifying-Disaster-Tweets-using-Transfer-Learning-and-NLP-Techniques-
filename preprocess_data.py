import pandas as pd

# Load the training and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Print the first 5 rows of the training dataset
print(train_df.head())

# Preprocess the data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs from text
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuation from text
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers from text
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace from text
    text = ' '.join(text.split())
    
    return text

# Apply the preprocessing function to the 'text' column of the training and test datasets
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Print the first 5 rows of the preprocessed training dataset
print(train_df.head())
