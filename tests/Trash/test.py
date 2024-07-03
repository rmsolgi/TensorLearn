import tensorflow as tf

# Define the number of dimensions
num_dim_x = 4

# Create a range of axis indices
axes = tf.range(num_dim_x)
axes+=2
# Print the axes
print("Axes:")
print(axes)