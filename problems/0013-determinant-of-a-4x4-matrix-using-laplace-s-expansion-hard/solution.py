import torch

def get_minors(matrix,row,col):
  minor=[]
  for i in range(len(matrix)):
    if i==row:
      continue
    new_row=[]
    for j in range(len(matrix[0])):
      if j==col:
        continue
      new_row.append(matrix[i][j])
    minor.append(new_row)
  return minor

def determinant_4x4(matrix):
  if len(matrix)==1:
    return matrix[0][0]
  elif len(matrix)==2:
    return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
  determinant_value=0
  for col in range(len(matrix)):
    sign=(-1)**col
    element=matrix[0][col]
    minor=get_minors(matrix,0,col)
    determinant_value=determinant_value+(sign*element*determinant_4x4(minor))
  return determinant_value.item()


