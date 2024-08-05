import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
 
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df
 
def dimensions(A):
    dimensionality = np.linalg.matrix_rank(A)
    print(f"Dimensionality of the vector space: {dimensionality}")
 
def count_vectors(A):
    num_vectors = A.shape[0]
    print(f"Number of vectors in this vector space: {num_vectors}")
 
def rank_of_matrix(A):
rank_A = np.linalg.matrix_rank(A)
    print(f"Rank of Matrix A: {rank_A}")
    
def compute_costs(A, C):
    # Compute the pseudo-inverse of A
    A_pseudo_inv = np.linalg.pinv(A)
    # Calculate the model vector X
    X = np.dot(A_pseudo_inv, C)
    return X
    
def classify_customers(df):
    df['Class'] = np.where(df['Payment (Rs)'] > 200, 'RICH', 'POOR')
X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    y = df['Class'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("
Classification Report:
print(classification_report(y_test, y_pred))
def main():
    file_path = r"C:\Users\year3\Downloads\Lab Session Data (1).xlsx"
    sheet_name = 'Purchase data'
    df = load_data(file_path, sheet_name)
    
    columns_to_print = df[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']]
    print("Selected columns:")
    print(columns_to_print)

    A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = df[['Payment (Rs)']].values

    print("
Matrix A:")
    print(A)
    print("
Matrix C:")
    print(C)

    dimensions(A)
    count_vectors(A)
    rank_of_matrix(A)

    # Calculate costs with the pseudo inverse
    X = compute_costs(A, C)
    print("
Cost of each product in stock for sale:")
    print(X)

classify_customers(df)

if __name__ == "__main__":
    main()
