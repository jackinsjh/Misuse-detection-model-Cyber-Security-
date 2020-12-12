import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")  # ignore unnecessary warnings



def main():
    def clean_dataset(df):  # removing invalid values
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep]



    # read dataset, using delimiter ','
    data = pd.read_csv('sample.csv', delimiter=',')



    # Data Inspection
    # these dataset contains some useful attributes for given objectives
    # metadata provided
    print('-------------------')
    print('List of initial attributes of the dataset\n')
    print(data.columns)
    print('-------------------')



    # before replacing it with some default value, conduct Feature Removal (from a Table)
    # Feature Removal - removing unnecessary features for the analysis
    exceptionAttributes = []  # list of attributes to remove
    exceptionAttributes.append('Dst Port')
    exceptionAttributes.append('Protocol')
    exceptionAttributes.append('Timestamp')
    exceptionAttributes.append('Flow Duration')
    exceptionAttributes.append('Flow ID')
    exceptionAttributes.append('Src IP')
    exceptionAttributes.append('Src Port')
    exceptionAttributes.append('Dst IP')

    for i in data.columns:
        if i in exceptionAttributes:
            data.drop(i, axis=1, inplace=True)


    # remove rows with invalid value
    data = data.dropna()
    data = clean_dataset(data)


    # using 'web attack' data as unlabeled data
    unlabData = data[data.Label == 'Web Attacks']  # data with 'Web Attacks' -> treated as unlabeled data
    data = data[data.Label != 'Web Attacks']




    # Feature Engineering
    # Label-Encoding categorical value of 'Label' column
    le = LabelEncoder()
    data['Label'] = le.fit_transform(data['Label'])



    # Process - Data Analysis / Evaluation

    # split dataset for the set of labels and the set of data
    y = data['Label']  # Label Data
    xData = data.drop('Label', axis=1)  # Data without the label

    # Spliting dataset
    x_train, x_test, y_train, y_test = train_test_split(xData, y, test_size=0.2, shuffle=True, random_state=1004)

    # define several knn classifiers with different parameters
    knnParameters = [
        {'n_neighbors': [3, 5, 7], 'metric': ['minkowski', 'euclidean']},
    ]  # parameter list
    knn = KNeighborsClassifier()
    # define several knn classifiers with different parameters
    knnGrid = GridSearchCV(knn, knnParameters, cv=2)  # cv=2 -> performs 2-fold cross validation
    knnGrid.fit(x_train, y_train)  # fit the multiple grid models
    print('KNN algorithm')
    print('# best parameter set')
    print(knnGrid.best_params_)
    print('# best score')
    print(knnGrid.best_score_)
    print('# score of each model')
    print(knnGrid.cv_results_['mean_test_score'])
    print('-------------------')

    # confusion matrix
    y_pred = knnGrid.predict(x_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat = pd.DataFrame(conf_mat, columns=["Benign", "Botnet", "Brute", "DoS", "DDoS", "Infil"],
                  index=["Benign", "Botnet", "Brute", "DoS", "DDoS", "Infil"])
    sn.heatmap(conf_mat, annot=True, fmt='d')
    plt.title("Confusion Matrix - left: Actual, bottom: Predicted")
    plt.show()



    # Applying unlabeled data (web attack) on the KNN model

    xDataUnlab = unlabData.drop('Label', axis=1)  # unlabeled data without the label

    # prediction of the knn model
    y_pred_unlab = knnGrid.predict(xDataUnlab)
    print('\n\n\n')
    print('Prediction result of the model with unlabeled(web attack) dataset')
    print('- Benign: {}'.format(np.count_nonzero(y_pred_unlab == 0)))
    print('- Botnet: {}'.format(np.count_nonzero(y_pred_unlab == 1)))
    print('- Brute-force attack: {}'.format(np.count_nonzero(y_pred_unlab == 2)))
    print('- Denial-of-Service: {}'.format(np.count_nonzero(y_pred_unlab == 3)))
    print('- Distributed Denial-of Service: {}'.format(np.count_nonzero(y_pred_unlab == 4)))
    print('- Infilteration: {}'.format(np.count_nonzero(y_pred_unlab == 5)))

main()