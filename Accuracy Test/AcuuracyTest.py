import math



# Generate random average point coordinates
avg_x, avg_y = 148,	460.2



# Generate random ground truth point coordinates
gt_x, gt_y =58	,447

# Calculate Euclidean distance
distance = math.sqrt((avg_x - gt_x)**2 + (avg_y - gt_y)**2)

# Set accuracy threshold
threshold = 200

# Compare distance with threshold and calculate accuracy
if distance <= threshold:
    accuracy = 100 - (distance / threshold) * 100
else:
    accuracy = 0

print("Ground truth point coordinates:", gt_x, gt_y)
print("Average point coordinates:", avg_x, avg_y)
print("Distance:", distance)
print("Accuracy:", accuracy)
