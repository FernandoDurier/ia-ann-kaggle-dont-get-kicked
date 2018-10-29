#References
#Karimi, R. and Gero, Z. (n.d.). Don’t Get Kicked: Predict if a Car Purchased at Auction is Lemon. [ebook] Available at: http://www.mathcs.emory.edu/~rkarimi/files/dontgetkicked.pdf [Accessed 29 Oct. 2018].

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

le = preprocessing.LabelEncoder()

#Function to remove NaN correctly both for numerical and categorical data
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

#Loading Data from CSV
#Separating into training and test datasets
#RefId,IsBadBuy,PurchDate,Auction,VehYear,VehicleAge,Make,Model,Trim,SubModel,Color,Transmission,WheelTypeID,WheelType,VehOdo,Nationality,Size,TopThreeAmericanName,MMRAcquisitionAuctionAveragePrice,MMRAcquisitionAuctionCleanPrice,MMRAcquisitionRetailAveragePrice,MMRAcquisitonRetailCleanPrice,MMRCurrentAuctionAveragePrice,MMRCurrentAuctionCleanPrice,MMRCurrentRetailAveragePrice,MMRCurrentRetailCleanPrice,PRIMEUNIT,AUCGUART,BYRNO,VNZIP1,VNST,VehBCost,IsOnlineSale,WarrantyCost
data = pd.read_csv('./resources/training.csv')

X_train, X_test, y_train, y_test = train_test_split(data, data['IsBadBuy'], test_size=0.3)

clean_training_set = pd.DataFrame(X_train)
clean_test_set = pd.DataFrame(X_test)

# print("training features: \n", X_train.head(), '\n')
# print("training target: \n", y_train.head(), '\n')
# print("test features: \n", X_test.head(), '\n')
# print("test target: \n", y_test.head(), '\n')
#Removing Irrelevnat Attributes
#Removing Unique IDs
clean_training_set.loc[:, clean_training_set.columns != 'RefId']
clean_test_set.loc[:, clean_test_set.columns != 'RefId']

#Removing Much NaN fields or fields with repeated data, as did in [1]
#Removing fields containing most of all NaN
clean_training_set.loc[:, clean_training_set.columns != 'PRIMEUNIT']
clean_test_set.loc[:, clean_test_set.columns != 'PRIMEUNIT']

clean_training_set.loc[:, clean_training_set.columns != 'AUCGUARD']
clean_test_set.loc[:, clean_test_set.columns != 'AUCGUARD']

#Removing repeated information
clean_training_set.loc[:, clean_training_set.columns != 'VehYear'] #repeated info from VehAge
clean_test_set.loc[:, clean_test_set.columns != 'VehYear'] #repeated info from VehAge

clean_training_set.loc[:, clean_training_set.columns != 'WheelType'] #repeated info from WheelTypeId
clean_test_set.loc[:, clean_test_set.columns != 'WheelType'] #repeated info from WheelTypeId

clean_training_set.loc[:, clean_training_set.columns != 'VNZIP1'] #repeated info from VNST
clean_test_set.loc[:, clean_test_set.columns != 'VNZIP1'] #repeated info from VNST

#inferred the rest of data that could be empty/NaN
clean_training_set = DataFrameImputer().fit_transform(clean_training_set)
clean_test_set = DataFrameImputer().fit_transform(clean_test_set)

#encoding categorical features for training set
summary_of_features_by_types = clean_training_set.dtypes == object
for key, value in summary_of_features_by_types.items():
    if value:
        clean_training_set[key] = clean_training_set[key].astype('category').cat.codes
    if key == 'IsBadBuy':
        clean_training_set[key] = clean_training_set[key].astype('category').cat.codes

#encoding categorical features for test set
summary_of_features_by_types_test = clean_test_set.dtypes == object
for key, value in summary_of_features_by_types_test.items():
    if value:
        clean_test_set[key] = clean_test_set[key].astype('category').cat.codes
    if key == 'IsBadBuy':
        clean_training_set[key] = clean_training_set[key].astype('category').cat.codes

#Instantiating ANN
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,20,10,5), random_state=1)
#Training our model
clf.fit(clean_training_set, y_train)
#Testing the trained model against the test dataset
classifications = clf.predict(clean_test_set)

# print("weights between input and first hidden layer:")
# print(clf.coefs_[0])
# print("\nweights between first hidden and second hidden layer:")
# print(clf.coefs_[1])
# print("\nweights between second hidden and third hidden layer:")
# print(clf.coefs_[2])

print("Classification: \n", classifications)
print("Confusion Matrix: \n", confusion_matrix(y_test,classifications))
print("Classification Report: \n", classification_report(y_test,classifications))