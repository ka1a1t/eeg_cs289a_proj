import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn import svm


from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

# # ensure only cpu tf is run
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# # Hide GPU from visible devices
# tf.config.set_visible_devices([], 'GPU')


import warnings
warnings.filterwarnings("ignore")

########## Set seed and aesthetics for plots
rand_seed_int = 7739
sns.set(style="whitegrid", color_codes=True)


########## Data Loading and Processing
path_to_data = "./Dataset/"  # generate paths
file_names = ["user_a.csv", "user_b.csv", "user_c.csv", "user_d.csv"]
data_paths = [path_to_data + fn for fn in file_names]


user_dfs = [pd.read_csv(data_path) for data_path in data_paths]

for df, letter in zip(user_dfs, "ABCD"):
    df['user'] = pd.Series(letter, index=df.index) # append categorical variable denoting user

df = pd.concat(user_dfs, axis=0)

# shuffle and reindex dataframe
df = df.sample(frac=1, random_state=rand_seed_int).reset_index(drop=True)



def data_process(df):
    # generate one hot encoded dummy variables for user categories
    user_dummies = pd.get_dummies(df['user'], prefix='user')

    # user_dummies
    # concatenate dummies and drop orig user feature
    df = pd.concat([df, user_dummies], axis=1)
    df = df.drop('user', axis=1)
    
    # separate target feature class from data
    y = df['Class'].copy()
    X = df.drop('Class', axis=1)
    
    # dummy columns to not transform
    cols_to_std = list(X.columns[:-4])
    
    # cast target as integer
    y = y.astype(int)
    
    # split into training and testing sets: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=rand_seed_int)
    
    # decided to use k-fold cv rather than leave-one-out cross-validation
    # I think k-fold makes more sense with respect to the # of observations
    # split training into training and validation sets: 70% train, 30% validation
    # X_train, X_val, y_train, y_val = train_test_split(X_Train, y_Train, train_size=0.7, random_state=rand_seed_int)
    
    # scale frequencies s.t. features are normalized and dummies are left alone

    ct = ColumnTransformer([
            ('blah_standard_scaler', StandardScaler(), cols_to_std)
        ], remainder='passthrough')
    
    ct.fit(X_train)
    # return as dataframes
    X_train = pd.DataFrame(ct.transform(X_train), columns=X.columns)
    # X_val   = pd.DataFrame(ct.transform(X_val), columns=X.columns)
    X_test  = pd.DataFrame(ct.transform(X_test), columns=X.columns)
    
    # print(ct.named_transformers_)
    
    # return X_train, y_train, X_val, y_val, X_test, y_test
    return X_train, y_train, X_test, y_test


# X_train, y_train, X_val, y_val, X_test, y_test = data_process(df)
X_train, y_train, X_test, y_test = data_process(df)


########## LDA Plot Generation

param_grid = {
    'shrinkage': [1.0, 0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    'solver': ['lsqr']
}

gs = GridSearchCV(
    LDA(),
    param_grid=param_grid,
    verbose=1,
    cv=5,
    n_jobs=-1,
    return_train_score = True
)

lda_gs_results = gs.fit(X_train, y_train)

lda_train_scores = lda_gs_results.cv_results_['mean_train_score']
lda_val_scores = lda_gs_results.cv_results_['mean_test_score']
lda_shrnks = [pdict['shrinkage'] for pdict in lda_gs_results.cv_results_['params']]


fig, ax = plt.subplots()
ax.set_title('LDA Training & Validation Accuracies')
ax.plot(np.log10(lda_shrnks), lda_train_scores, color='teal', label='Training Accuracies')
ax.plot(np.log10(lda_shrnks), lda_val_scores, color='purple', label='Validation Accuracies')
ax.set_xlabel("log_10(shrinkage)")
ax.set_ylabel("Accuracy")
ax.legend()
plt.savefig("script_lda_train_vs_val_accuracy.png")


########## QDA Plot Generation

param_grid = {
    'reg_param': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 2.0],
    'tol': [0.0001]
}

gs = GridSearchCV(
    QDA(),
    param_grid=param_grid,
    verbose=1,
    cv=5,
    n_jobs=-1,
    return_train_score = True
)

qda_gs_results = gs.fit(X_train, y_train)

qda_train_scores = qda_gs_results.cv_results_['mean_train_score']
qda_val_scores = qda_gs_results.cv_results_['mean_test_score']
qda_reg_param = [pdict['reg_param'] for pdict in qda_gs_results.cv_results_['params']]

fig, ax = plt.subplots()
ax.set_title('QDA Training & Validation Accuracies')
ax.plot((qda_reg_param), qda_train_scores, color='teal', label='Training Accuracies')
ax.plot((qda_reg_param), qda_val_scores, color='purple', label='Validation Accuracies')
ax.set_xlabel("reg_param")
ax.set_ylabel("Accuracy")
ax.legend()
plt.savefig("script_qda_train_vs_val_accuracy.png")


########## KNC Plot Generation

param_grid = {
    'n_neighbors': [ 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'weights': ['distance'],
    'metric': ['manhattan']
}

gs = GridSearchCV(
    KNC(),
    param_grid=param_grid,
    verbose=1,
    cv=5,
    n_jobs=-1,
    return_train_score = True

)

knc_gs_results = gs.fit(X_train, y_train)

knc_gs_results = gs.fit(X_train, y_train)

knc_train_scores = knc_gs_results.cv_results_['mean_train_score']
knc_val_scores = knc_gs_results.cv_results_['mean_test_score']
knc_n_nghb = [pdict['n_neighbors'] for pdict in knc_gs_results.cv_results_['params']]

fig, ax = plt.subplots()
ax.set_title('KNC Training & Validation Accuracies')
ax.plot((knc_n_nghb), knc_train_scores, color='teal', label='Training Accuracies')
ax.plot((knc_n_nghb), knc_val_scores, color='purple', label='Validation Accuracies')
ax.set_xlabel("Neighbors")
ax.set_ylabel("Accuracy")
ax.legend()
plt.savefig("script_knc_train_vs_val_accuracy.png")

##########  Fully Connected Neural Network Plot Generation


X_Train = X_train.copy()

# create validation set for FCNN training
X_train_fc ,X_val_fc, y_train_fc, y_val_fc = train_test_split(X_Train, y_train, train_size=0.8, 
                                                              random_state=rand_seed_int)

print("X_train_fc shape", X_train_fc.shape)
print("X_val_fc shape", X_val_fc.shape)


# 2 hidden layers, 150 channels, output channel 3 nodes
# activation is relu on hidden chnls, softmax at output nodes
# uniform used for initialization
model_fc = Sequential()
model_fc.add(Dense(150, input_dim=116, activation="relu", kernel_initializer="uniform"))
model_fc.add(Dense(150, activation="relu", kernel_initializer="uniform"))
model_fc.add(Dense(3, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)


model_fc.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=opt,
    metrics=['accuracy']
)


hist_model_fc = model_fc.fit(
                    X_train_fc.values,
                    y_train_fc.values,
                    epochs=40,
                    validation_data=(X_val_fc.values, y_val_fc.values)
                    )

fcnn_train_scores = hist_model_fc.history['accuracy']
fcnn_val_scores = hist_model_fc.history['val_accuracy']
fcnn_epochs = list(range(1, 41))

fig, ax = plt.subplots()
ax.set_title('FCNN Training & Validation Accuracies')
ax.plot((fcnn_epochs), fcnn_train_scores, color='teal', label='Training Accuracies')
ax.plot((fcnn_epochs), fcnn_val_scores, color='purple', label='Validation Accuracies')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.legend()
plt.savefig("script_fcnn_train_vs_val_accuracy.png")


########## SVC Plot Generation

param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [0.1],
              'kernel': ['rbf']}
gs = GridSearchCV(
    svm.SVC(),
    param_grid, 
    refit = True, 
    verbose = 10,
    return_train_score = True,
    n_jobs = -1
)

svc_gs_results = gs.fit(X_train,y_train)

svc_train_scores = svc_gs_results.cv_results_['mean_train_score']
svc_val_scores = svc_gs_results.cv_results_['mean_test_score']
svc_Cs = [pdict['C'] for pdict in svc_gs_results.cv_results_['params']]

fig, ax = plt.subplots()
ax.set_title('SVC Training & Validation Accuracies')
ax.plot(np.log10(svc_Cs), svc_train_scores, color='teal', label='Training Accuracies')
ax.plot(np.log10(svc_Cs), svc_val_scores, color='purple', label='Validation Accuracies')
ax.set_xlabel("log_10(C)")
ax.set_ylabel("Accuracy")
ax.legend()
plt.savefig("script_svc_train_vs_val_accuracy.png")
