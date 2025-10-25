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
from matplotlib.backends.backend_pdf import PdfPages


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

columns = ['algorithm_name','train_or_test_data','accuracy','precision','recall','F1','AUC']

def get_train_test_data(file_name, split_value_factor):
    np.random.seed(42)
    # Read data from Excel file
    df = pd.read_csv(file_name)
    # Define your features (X) and target (y)
    # Replace 'target_column' with the name of your target variable
    X = df[['x1','x2']].values
    y = df['y'].values
    # Split the data into training and testing sets.
    # 80% for training and 20% for testing is a common split (test_size=0.2).
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_value_factor, random_state=42)
    return X,y,x_train, x_test, y_train, y_test


def fit_to_logistic_regression(algorithm, x_train, x_test, y_train, y_test, rows, subplot_count, data):
    algorithm.fit(x_train, y_train)
    # Make predictions on the test set
    y_test_pred = algorithm.predict(x_test)
    y_test_pred_prob = algorithm.predict_proba(x_test)
    
    y_train_pred = algorithm.predict(x_train)
    y_train_pred_prob = algorithm.predict_proba(x_train)
    
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred, average='micro')
    recall_train = recall_score(y_train, y_train_pred, average='micro')
    f1score_train = f1_score(y_train, y_train_pred, average='micro')
    roc_train = roc_auc_score(y_train, y_train_pred_prob, multi_class='ovr')
    data['train_or_test_data'].append('train')
    data['accuracy'].append(accuracy_train)
    data['precision'].append(precision_train)
    data['recall'].append(recall_train)
    data['F1'].append(f1score_train)
    data['AUC'].append(roc_train)
    print(f" -->> TRAIN -->> Accuracy: {accuracy_train:.4f} |Precision: {precision_train:.4f} |Recall: {recall_train:.4f} |F1_score: {f1score_train:.4f} |ROC: {roc_train:.4f}")
     # Evaluate the model
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='micro')
    recall_test = recall_score(y_test, y_test_pred, average='micro')
    f1score_test = f1_score(y_test, y_test_pred, average='micro')
    roc_test = roc_auc_score(y_test, y_test_pred_prob, multi_class='ovr')
    data['train_or_test_data'].append('test')
    data['accuracy'].append(accuracy_test)
    data['precision'].append(precision_test)
    data['recall'].append(recall_test)
    data['F1'].append(f1score_test)
    data['AUC'].append(roc_test)
    print(f" -->> TEST -->> Accuracy: {accuracy_test:.4f} |Precision: {precision_test:.4f} |Recall: {recall_test:.4f} |F1_score: {f1score_test:.4f} |ROC: {roc_test:.4f}")
    

    # avg_precision = average_precision_score(y_test, y_test_pred)
    # print(f"Average Precision: {avg_precision:.4f}")
    # Detailed performance report
    # aucValue = auc(y_test, y_test_pred)
    # print(f"AUC: {aucValue:.4f}") 
    
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)  
    y_train_bin = label_binarize(y_train, classes=classes)
    
    plt.subplot(rows, 3, subplot_count)
    plot_decision_boundary(algorithm, None, x_train, y_train, title=algorithm_name)
    subplot_count+=1
    plt.tight_layout()
    plt.subplot(rows,3,subplot_count)
    subplot_count+=1
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_train_bin[:, i], y_train_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('ROC Curve Train Data - OvR')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.subplot(rows,3,subplot_count)
    subplot_count+=1
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('ROC Curve Test Data - OvR')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return subplot_count

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

def generate_logistic_regression_plot(algorithm_name, algorithm, x_train, x_test, y_train, y_test, rows, subplot_count, data):
    print(f" -->> Starting ML Model fit for {algorithm_name}")
    data['algorithm_name'].append(algorithm_name)
    data['algorithm_name'].append(algorithm_name)
    subplot_count=fit_to_logistic_regression(algorithm, x_train, x_test, y_train, y_test, rows, subplot_count, data)
    print(f" -->> ML Model fit completed for {algorithm_name}")
    scaler = StandardScaler().fit(x_train)
    return subplot_count

    

if __name__ == "__main__":
    with pd.ExcelWriter('Logistic_Reg_Output.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        for file in files:
            data = {
                'algorithm_name': [],
                'train_or_test_data': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'F1': [],
                'AUC': [],
            }
            X, y, x_train, x_test, y_train, y_test = get_train_test_data(file_name=file, split_value_factor=0.3)
            classes = np.unique(y)
            class_names = [str(c) for c in classes]
            print("Values of y are classified as:", class_names)
            plt.figure(figsize=(30,30))
            rows = len(algorithms)
            subplot_count = 1
            for algorithm_name, algorithm in algorithms.items():
                # plt.subplot(rows, 3, 1)
                # plot_scatter(x_train, y_train, classes, "Scatter (Train) (X1 vs X2)")
                # plt.tight_layout()
                # plt.subplot(rows, 3, 4)
                # plot_scatter(x_test, y_test, classes, "Scatter (Test) (X1 vs X2)")
                # plt.tight_layout()

                # plt.subplot(rows, 3, 2)
                # plot_decision_boundary(algorithm, None, x_test, y_test, title="Decision boundary (Test)")
                # plt.tight_layout()
                subplot_count = generate_logistic_regression_plot(algorithm_name, algorithm, x_train, x_test, y_train, y_test, rows, subplot_count, data)
            plt.savefig(file+'.pdf')
            plt.close()
            
            df_new = pd.DataFrame(data)
            df_new.to_excel(writer, sheet_name=file, index=False)
    print("✅ Data written to File")