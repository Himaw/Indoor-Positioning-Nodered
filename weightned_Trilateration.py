# from scipy.optimize import minimize
# import numpy as np

# def trilateration(positions, distances, weights):
#     # Define the objective function for optimization
#     def objective_function(point):
#         x, y = point
#         return np.sum(weights * ((positions[:, 0] - x) ** 2 + (positions[:, 1] - y) ** 2 - distances ** 2) ** 2)

#     # Initial guess for the tag position
#     initial_guess = np.mean(positions, axis=0)

#     # Perform the optimization
#     result = minimize(objective_function, initial_guess, method='Nelder-Mead')

#     # Return the estimated tag position
#     return result.x

# # Example usage
# anchors = np.array([(0, 0), (835, 0), (0, 665)])  # Anchor positions
# distances = np.array([800, 820, 580])  # Distance measurements
# weights = np.array([1.0, 0.8, 0.6])  # Weights for distance measurements

# estimated_position = trilateration(anchors, distances, weights)
# print("Estimated position:", estimated_position)


import numpy as np

def trilateration(positions, distances):
    # Get the number of anchors
    num_anchors = len(positions)

    # Construct the matrix A and vector b for the linear system of equations
    A = np.zeros((num_anchors - 1, 2))
    b = np.zeros(num_anchors - 1)

    

    for i in range(num_anchors - 1):
        A[i, 0] = 2 * (positions[i + 1, 0] - positions[0, 0])
        A[i, 1] = 2 * (positions[i + 1, 1] - positions[0, 1])
        b[i] = distances[0]**2 - distances[i + 1]**2 - positions[0, 0]**2 + positions[i + 1, 0]**2 - positions[0, 1]**2 + positions[i + 1, 1]**2
   
    # Solve the linear system of equations
    result = np.linalg.lstsq(A, b, rcond=None)

    # Calculate the estimated position
    x = result[0][0]
    y = result[0][1]

    return x, y

# Example usage
anchors = np.array([(0, 0), (835, 0), (0, 665)])  # Anchor positions
distances = np.array([343, 597, 997])  # Distance measurements
# distances = np.array([343, 597, 897])  # Distance measurements


estimated_position = trilateration(anchors, distances)
print("Estimated position:", estimated_position)
