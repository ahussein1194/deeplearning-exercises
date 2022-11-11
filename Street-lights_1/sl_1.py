import numpy as np

weights = np.array([0.5, 0.48, -0.7])

alpha = 0.1

streetlights = np.array([
                          [1, 0, 1],
                          [0, 1, 1],
                          [0, 0, 1],
                          [1, 1, 1],
                          [0, 1, 1],
                          [1, 0, 1]
                          ])

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

# Training over one example only.
#input = streetlights[0]
#goal_pred = walk_vs_stop[0]
#for iteration in range(20):
#    pred = input.dot(weights)
#    error = (pred - goal_pred) ** 2
#    delta = pred - goal_pred
#    weights = weights - ((input * delta) * alpha)
#    print("Prediction: " + str(pred))
#    print("Error: " + str(error))

# Training over the entire dataset - stochastic gradient descent.
#for iteration in range(40):
#    print(f'Weights: {weights}')
#    error_for_all_lights = 0
#    for i in range(len(walk_vs_stop)):
#        input = streetlights[i]
#        goal_pred = walk_vs_stop[i]
#        pred = input.dot(weights)
#        error = (pred - goal_pred) ** 2
#        error_for_all_lights += error
#        delta = pred - goal_pred
#        weights = weights - ((input * delta) * alpha)
#        print(f'Prediction for input #{i+1}: {pred}' )
#        print(f'Error for input #{i+1}: {error}')
#        print(f'Delta for input #{i+1}: {delta}')
#    print(f'Total error for iteration #{iteration+1}: {error_for_all_lights}')
#    print()

# Training over the entire dataset - full/average gradient descent.
for iteration in range(1000):
    print(f'Weights: {weights}')
    error_for_all_lights = 0
    total_weight_delta = 0
    average_wt_delta = 0
    for i in range(len(walk_vs_stop)):
        input = streetlights[i]
        goal_pred = walk_vs_stop[i]
        pred = input.dot(weights)
        error = (pred - goal_pred) ** 2
        error_for_all_lights += error
        delta = pred - goal_pred
        weight_delta = ((input * delta) * alpha)
        total_weight_delta += weight_delta
        print(f'Prediction for input #{i+1}: {pred}' )
        print(f'Error for input #{i+1}: {error}')
        print(f'Delta for input #{i+1}: {delta}')
    average_wt_delta = total_weight_delta / len(walk_vs_stop)
    weights = weights - average_wt_delta
    print(f'Total error for iteration #{iteration+1}: {error_for_all_lights}')
    print()
