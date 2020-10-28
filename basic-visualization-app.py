import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style= "ticks")
plt.style.use("fivethirtyeight")
from sklearn.metrics import *
from sklearn.decomposition import PCA

st.title("Streamlit Example")
# st.set_option('deprecation.showPyplotGlobalUse', False)

st.write(
    """
    # Explore different classifier and datasets
	Which one is the best?
    """
)

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(name):

    if name == "Iris":
        data = datasets.load_iris()

    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    
    else:
    	data = datasets.load_wine()
    
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)

st.write(f"### Shape of dataset {X.shape}")
st.write(f"### No. of classes {len(np.unique(y))}")

def add_parameter_ui(clf_name):
	params = dict()
	
	if clf_name == "KNN":
		K = st.sidebar.slider("K", 1, 15)
		params["K"] = K
	
	elif clf_name == "SVM":
		C = st.sidebar.slider("C", 0.1, 10.0)
		params["C"] = C
	
	else:
		max_depth = st.sidebar.slider("max_depth", 2, 15)
		n_estimators = st.sidebar.slider("n_estimators", 1, 100)
		params["max_depth"] = max_depth
		params["n_estimators"] = n_estimators
	return params
	
params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
	if clf_name == "KNN":
		clf = KNeighborsClassifier(n_neighbors=params["K"])

	elif clf_name == "SVM":
		clf = SVC(C=params["C"])

	else:
		clf = RandomForestClassifier(n_estimators=params["n_estimators"], 
									 max_depth=params["max_depth"], random_state=42)

	return clf

clf = get_classifier(classifier_name, params)

#### CLASSIFICATION ####
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

def print_results(y_test, y_pred, model='Results:'):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    if len(np.unique(y)) > 2:
        st.write(f"Model: {model}")
        st.write(f"Accuracy is {round(accuracy_score(y_test, y_pred), 2)}")
        st.write(f"\nclassification Report:\n {classification_report(y_test, y_pred)}\n")
    else:
        st.write(f"Model: {model}")
        st.write(f"Accuracy is {round(accuracy_score(y_test, y_pred), 2)}")
        st.write(f"\nPrecision-score is {round(precision_score(y_test, y_pred), 2)}")
        st.write(f"\nRecall-score is {round(recall_score(y_test, y_pred), 2)}")
        st.write(f"\nF1-score is {round(f1_score(y_test, y_pred), 2)}")
        st.write(f"\nclassification Report:\n {classification_report(y_test, y_pred)}\n")

print_results(y_test, y_pred, model=f'{classifier_name}')

# acc = accuracy_score(y_test, y_pred)
# st.write(f"Accuracy = {acc}")

# Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
sns.scatterplot(x=x1, y=x2, hue=y, palette="plasma")
# plt.scatter(x1, x2, c=y, cmap="plasma", alpha=0.8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
# plt.colorbar()
st.pyplot(fig)