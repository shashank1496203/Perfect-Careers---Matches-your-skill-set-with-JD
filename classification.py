import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def preprocess_text(text, regex):
    # List for storing filtered sentences
    data = []

    # For sentence in sentences:
        # remove target value from predictor and lowercase the text
    text=re.sub(f'{regex}[a-z]+', '', text.lower(), re.I)
    # split into words
    word_tokens = word_tokenize(text)

    # Stemming of words
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in word_tokens]

    # Remove punctuation from word tokens
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in stemmed]

    # Check if all the tokens are alphanumeric
    words = [word for word in stripped if word.isalpha()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_text = [w for w in words if not w in stop_words]
    return " ".join(filtered_text)

def create_dataset(files):
    df = pd.DataFrame() 
    mapping = {
        'Software_Engineer': ('software eng',0), 
        'Data_Engineer': ('data eng',1), 
        'Data_Scientist': ('data sci',2)
    }
    for file in files:
        name = file.strip('.csv')
        temp = pd.read_csv(file)
        temp['Summary'] = temp['Summary'].map(lambda text: preprocess_text(text, mapping[name][0]))
        temp = temp.assign(Job_title=mapping[name][1])
        df = df.append(temp, ignore_index = True)
    matrix = TfidfVectorizer(max_features=75)
    return matrix.fit_transform(df['Summary']).toarray(),df.iloc[:,0]

files = ['Software_Engineer.csv', 'Data_Engineer.csv', 'Data_Scientist.csv'] #Load all csv files
x,y = create_dataset(files)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42) #Split data into testing and training

################################################################################################################################################################

#Random Forest Grid Search
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 
param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
print ("Best params of RandomForestClassfier: ",CV_rfc.best_params_)

################################################################################################################################################################

#Extra Trees Classifier 
etc = ExtraTreesClassifier(max_depth=None)

param_grid = { 
    'n_estimators': [500, 700],
    'min_samples_split': [2,5],
    'criterion': ["gini", "entropy"],
    'max_features': ['auto', 'sqrt']
}

CV_etc = GridSearchCV(estimator=etc, param_grid=param_grid, cv= 5)
CV_etc.fit(X_train, y_train)
print ("Best params of ExtraTreesClassifier: ", CV_etc.best_params_)

################################################################################################################################################################

#SVC model
parameters = {'C': [1, 10], 
          'gamma': [0.001, 0.01, 1]}
model = SVC()
CV_svc = GridSearchCV(estimator=model, param_grid=parameters)
CV_svc.fit(X_train, y_train)
print("Best params of SVC: ", CV_svc.best_params_)

################################################################################################################################################################

#Voting Classifier (Using the best parameters found previously)

clf1 = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=2, criterion='entropy', max_features='auto')
clf2 = RandomForestClassifier(max_features='auto', n_estimators=700, n_jobs=-1,oob_score = True)
clf3 = make_pipeline(StandardScaler(), SVC(gamma='auto',  probability=True))
vc = VotingClassifier(weights=[2,2,1], estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
vc = vc.fit(X_train, y_train)
print(accuracy_score(vc.predict(X_test), y_test))