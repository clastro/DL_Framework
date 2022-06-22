import tensorflow as tf

def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = tf.Variable(x0)
	with GradientTape() as tape: # 연산과정을 tape에 기록
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x,x)
    # y = x*x
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))
