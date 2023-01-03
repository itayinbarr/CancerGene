import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def build_model():
    # Import labels
    y = pd.read_csv('./data/actual.csv')

    # Recode labels to numbers
    y = y.replace({'ALL': 0, 'AML': 1})

    # Import training data
    df_train = pd.read_csv('./data/data_set_ALL_AML_train.csv')

    # Import testing data
    df_test = pd.read_csv('./data/data_set_ALL_AML_independent.csv')

    # Remove "call" columns from training and testing data
    train_to_keep = [col for col in df_train.columns if "call" not in col]
    test_to_keep = [col for col in df_test.columns if "call" not in col]

    X_train_tr = df_train[train_to_keep]
    X_test_tr = df_test[test_to_keep]

    # Re-indexing data
    train_columns_titles = ['Gene Description', 'Gene Accession Number', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                            '10',
                            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
                            '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']

    X_train_tr = X_train_tr.reindex(columns=train_columns_titles)

    test_columns_titles = ['Gene Description', 'Gene Accession Number', '39', '40', '41', '42', '43', '44', '45', '46',
                           '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
                           '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72']

    X_test_tr = X_test_tr.reindex(columns=test_columns_titles)

    # Making each gene a feature, and every patient is a row
    X_train = X_train_tr.T
    X_test = X_test_tr.T

    # Clean up the gene description, and setting the gene accession
    # number as the feature labels

    # Clean up the column names for training and testing data
    X_train.columns = X_train.iloc[1]
    X_train = X_train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

    # Clean up the column names for Testing data
    X_test.columns = X_test.iloc[1]
    X_test = X_test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

    # Split into train and test (we first need to reset the index
    # as the indexes of two dataframes need to be the same before you combine them).

    # Subset the first 38 patients cancer types
    X_train = X_train.reset_index(drop=True)
    y_train = y[y.patient <= 38].reset_index(drop=True)

    # Subset the rest for testing
    X_test = X_test.reset_index(drop=True)
    y_test = y[y.patient > 38].reset_index(drop=True)

    # Convert from integer to float
    X_train_fl = X_train.astype(float, 64)
    X_test_fl = X_test.astype(float, 64)

    # Apply the same scaling to both datasets
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_fl)

    # Applying PCA
    # Searching for enough eigenvectors to represent 90% of variance
    pca = PCA()
    pca.fit_transform(X_train)

    total = sum(pca.explained_variance_)
    k = 0
    current_variance = 0
    while current_variance / total < 0.90:
        current_variance += pca.explained_variance_[k]
        k = k + 1

    # k represents it, now applying pca to preferred dimension
    pca = PCA(n_components=k)
    X_train = pca.fit(X_train_fl)
    X_train_pca = pca.transform(X_train_fl)
    X_test_pca = pca.transform(X_test_fl)

    log_grid = {'C': [1e-03, 1e-2, 1e-1, 1, 10],
                'penalty': ['l1', 'l2']}

    log_estimator = LogisticRegression(solver='liblinear')

    log_model = GridSearchCV(estimator=log_estimator,
                             param_grid=log_grid,
                             cv=3,
                             scoring='accuracy')

    log_model.fit(X_train_pca, y_train.iloc[:, 1])

    # Select best log model
    best_log = log_model.best_estimator_

    # Make predictions using the optimised parameters
    log_pred = best_log.predict(X_test_pca)

    print('Final Model Accuracy:', round(accuracy_score(y_test.iloc[:, 1], log_pred), 3))


build_model()
