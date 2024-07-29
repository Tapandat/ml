def construct_matrix(rows(columns,matrix):
  for i in range(rows):
    matrix_row = []
    for j in range(columns):
      element = int(input(f'Enter element for row {i+1} and column {j+1}: '))
      matrix_row.append(element)
    matrix.append(matrix_row)
def transposes_matrix(matrix,rows,columns,transpose_matrix):

  for i in range(columns):
    transpose_matrix_row = []
    for j in range(rows):
transpose_matrix_row.append(matrix[j][i])
    transpose_matrix.append(transpose_matrix_row)
rows = int(input('Enter number of rows: '))

نسا
columns = int(input('Enter number of columns: '))
matrix = []

construct_matrix(rows,columns,matrix)

print('Input Matrix: ')
for i in range(rows):
    
    for j in range(columns):
        print(matrix[i][j], end = " ")
    print()
transpose_matrix = []

 impose_matrix(matrix, rows, columns, transpose_matrix)

print('Transpose Matrix: ')
for i in range(rows):
for j in range(columns):
        print(transpose_matrix[i][j], end = " ")
    print()
