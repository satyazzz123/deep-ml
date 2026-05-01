def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    
    # 1. Check columns of 'a' against length of 'b'
    if len(a[0]) != len(b):
        return -1
        
    result = []
    
    # 2 & 3. No 'else' needed, added 'in' keyword
    for j in range(len(a)):
        row_sum = 0
        
        for i in range(len(b)):
            # 4. Use += to accumulate the sum
            row_sum += a[j][i] * b[i]

        # 5. Indented to be inside the outer loop (j), but outside the inner loop (i)
        result.append(row_sum)
        
    return result