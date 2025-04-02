import numpy as np

# Softmax function
def softmax(z):
    """
    Compute softmax values for each set of scores in z.
    
    Args:
        z: numpy array of shape (m, classes)
        
    Returns:
        softmax activations of shape (m, classes)
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Compute cost
def compute_cost(X, y, w, b):
    """
    Compute the cross-entropy cost for multi-class classification.
    
    Args:
        X: input data, shape (m, n)
        y: one-hot encoded labels, shape (m, classes)
        w: weights, shape (n, classes)
        b: bias, shape (classes,)
        
    Returns:
        cost: cross-entropy cost
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = softmax(z)
    cost = -np.sum(y * np.log(f_wb + 1e-8)) / m
    return cost

# Compute gradient
def compute_gradient(X, y, w, b):
    """
    Compute the gradient of the cost function.
    
    Args:
        X: input data, shape (m, n)
        y: one-hot encoded labels, shape (m, classes)
        w: weights, shape (n, classes)
        b: bias, shape (classes,)
        
    Returns:
        dj_dw: gradient with respect to w
        dj_db: gradient with respect to b
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = softmax(z)
    error = f_wb - y
    dj_dw = np.dot(X.T, error) / m
    dj_db = np.sum(error, axis=0) / m
    return dj_dw, dj_db

# Initialize weights 
def initialize_weights(input_size, output_size, method='random'):
    """
    Initialize model weights using different initialization methods.
    
    Args:
        input_size: number of input features
        output_size: number of output classes
        method: initialization method ('random', 'xavier')
        
    Returns:
        w: initialized weights
        b: initialized bias
    """
    if method == 'xavier':
        # Xavier/Glorot initialization
        w = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
    else:
        # Simple random initialization
        w = np.random.randn(input_size, output_size) * 0.01
    
    b = np.zeros(output_size)
    return w, b

# Evaluate model
def evaluate_model(X, y, w, b):
    """
    Evaluate the model and compute accuracy.
    
    Args:
        X: input data, shape (m, n)
        y: one-hot encoded labels, shape (m, classes)
        w: weights, shape (n, classes)
        b: bias, shape (classes,)
        
    Returns:
        accuracy: fraction of correctly classified examples
        y_pred: softmax predictions
    """
    z = np.dot(X, w) + b
    y_pred = softmax(z)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
    return accuracy, y_pred 