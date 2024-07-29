def is_multiplicable(matrix_a, matrix_b):
    return len(matrix_a[0]) == len(matrix_b)
def matrix_multiplication(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    cols_b = len(matrix_b[0])
    
    product_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
for j in range(cols_b):
            for k in range(cols_a):
                product_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return product_matrix

def main():
    matrix_a = [[1, 2, 3], [4, 5, 6]]
matrix_b = [[7, 8], [9, 10], [11, 12]]
    
    if is_multiplicable(matrix_a, matrix_b):
        product = matrix_multiplication(matrix_a, matrix_b)
        print("Product of matrices A and B:")
        for row in product:
            print(row)
else:
        print("Error: Matrices A and B can't be multiplied.")

if __name__ == "__main__":
    main()
