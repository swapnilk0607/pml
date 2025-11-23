import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, auc, recall_score,f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


files = {'clusters-4-v0.csv','clusters-4-v1.csv','clusters-4-v2.csv'}
poly_log_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # You can change degree
    ('scaler', StandardScaler()),  # Optional but recommended
    ('log_reg', LogisticRegression(multi_class='ovr',random_state=42, max_iter=2000))
])
algorithms = {
    'Logistic-Regression': LogisticRegression(multi_class='ovr',random_state=42, max_iter=2000),
    'Logistic-Regression-Poly': poly_log_reg,
    'SVC-Linear': SVC(kernel='linear', probability=True,random_state=42),
    'SVC-RBF': SVC(kernel='rbf', probability=True,random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'Neural-Network-5': MLPClassifier(hidden_layer_sizes=[5], max_iter=20000),
    'Neural-Network-5-5': MLPClassifier(hidden_layer_sizes=[5,5], max_iter=20000),
    'Neural-Network-5-5-5': MLPClassifier(hidden_layer_sizes=[5,5,5], max_iter=20000),
    'Neural-Network-10': MLPClassifier(hidden_layer_sizes=[10], max_iter=20000),
}

def get_train_test_split_values(file_name, split_value):
    np.random.seed(42)

    # Read data from Excel file
    df = pd.read_csv(file_name)
    # Define your features (X) and target (y)
    # Replace 'target_column' with the name of your target variable
    X = df[['x1','x2']].values
    y = df['y'].values
    # Split the data into training and testing sets.
    # 80% for training and 20% for testing is a common split (test_size=0.2).
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_value, random_state=42)
    return X,y,x_train, x_test, y_train, y_test


def fit_to_logistic_regression(algorithm, X, y, x_train, x_test, y_train, y_test):
    algorithm.fit(x_train, y_train)
    # Make predictions on the test set
    y_test_pred = algorithm.predict(x_test)
    y_test_pred_prob = algorithm.predict_proba(x_test)
    y_train_pred_prod = algorithm.predict_proba(x_train)
    print("y_test_pred_prod", y_test_pred_prob)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy: {accuracy:.4f}")
    precision = precision_score(y_test, y_test_pred, average='micro')
    print(f"Precision: {precision:.4f}")
    # avg_precision = average_precision_score(y_test, y_test_pred)
    # print(f"Average Precision: {avg_precision:.4f}")
    recall = recall_score(y_test, y_test_pred, average='micro')
    print(f"Recall: {recall:.4f}")    
    # Detailed performance report
    f1score = f1_score(y_test, y_test_pred, average='micro')
    print(f"F1_score: {f1score:.4f}") 
    # aucValue = auc(y_test, y_test_pred)
    # print(f"AUC: {aucValue:.4f}") 
    roc = roc_auc_score(y_test, y_test_pred_prob, multi_class='ovr')
    print(f"ROC: {roc:.4f}") 
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)  
    y_train_bin = label_binarize(y_train, classes=classes)
    
    plt.subplot(2,3,3)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_train_bin[:, i], y_train_pred_prod[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('ROC Curve (Train) - OvR')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.subplot(2,3,6)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('Test ROC Curve (Train) - OvR')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

# 2. EDA Plots (matplotlib only) and quick comments
def plot_scatter(X, y, classes, title):
    for label in classes:
        mask = y==label
        plt.scatter(X[mask,0], X[mask,1], label=f"class {label}", alpha=0.8)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title(title)
    
    
# Plot decision boundary function
def plot_decision_boundary(clf, scaler, X, y, title, gridstep=200, cmap_level=20):
    # We'll evaluate on feature space after optional scaling
    # For plotting, transform mesh grid with inverse_scaler if clf expects scaled input
    # Create mesh in original X1/X2 space
    x_min, x_max = X[:,0].min() - 1.0, X[:,0].max() + 1.0
    y_min, y_max = X[:,1].min() - 1.0, X[:,1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, gridstep), np.linspace(y_min, y_max, gridstep))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # If classifier expects scaled features (we used pipelines for scaled models in training), we must transform
    if scaler is not None:
        grid_in = scaler.transform(grid)
    else:
        grid_in = grid

    # For classifiers that give predict_proba with multi-columns or decision_function:
    try:
        Z = clf.predict(grid_in)
    except Exception as e:
        # maybe clf expects unscaled -> try raw grid
        Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, levels=cmap_level)
    # scatter points
    for label in np.unique(y):
        mask = y==label
        plt.scatter(X[mask,0], X[mask,1], label=f"class {label}", edgecolor='k')
    plt.xlim(x_min, x_max); 
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.xlabel("X1"); 
    plt.ylabel("X2")
    plt.legend()
    plt.tight_layout()

def generate_logistic_regression(algorithm_name, algorithm, file_name):
    print(f"ML Algorithm: {algorithm_name}")
    X, y, x_train, x_test, y_train, y_test = get_train_test_split_values(file_name=file_name, split_value=0.3)
    print("Training set shape:", x_train.shape, y_train.shape)
    print("Testing set shape:", x_test.shape, y_test.shape)
    classes = np.unique(y)
    class_names = [str(c) for c in classes]
    print("Values of y are classified as:", class_names)
    
    plt.figure(figsize=(10,7))
    plt.subplot(2, 3, 1)
    plot_scatter(x_train, y_train, classes, "Scatter (Train) (X1 vs X2)")
    plt.tight_layout()
    plt.subplot(2, 3, 4)
    plot_scatter(x_test, y_test, classes, "Scatter (Test) (X1 vs X2)")
    plt.tight_layout()
    fit_to_logistic_regression(algorithm, X, y, x_train, x_test, y_train, y_test)
    scaler = StandardScaler().fit(x_train)
    plt.subplot(2, 3, 2)
    plot_decision_boundary(algorithm, None, x_train, y_train, title="Decision boundary (Train)")
    plt.tight_layout()
    plt.subplot(2, 3, 5)
    plot_decision_boundary(algorithm, None, x_test, y_test, title="Decision boundary (Test)")
    plt.tight_layout()
    plt.savefig(algorithm_name+'-'+file_name+'.pdf')
    plt.close()
    

if __name__ == "__main__":
    
    for algorithm_name, algorithm in algorithms.items():
        for file in files:
            generate_logistic_regression(algorithm_name, algorithm, file)