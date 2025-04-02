import numpy as np
from mnist_model import compute_cost, compute_gradient, softmax

def initialize_adam(w, b, beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.001):
    """
    Initialize Adam optimizer parameters.
    
    Args:
        w: weights to optimize
        b: bias to optimize
        beta1: exponential decay rate for first moment estimates
        beta2: exponential decay rate for second moment estimates
        epsilon: small constant for numerical stability
        lr: learning rate
        
    Returns:
        Adam optimizer parameters
    """
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    return m_w, v_w, m_b, v_b, beta1, beta2, epsilon, lr, 0

def adam_update(w, b, dw, db, m_w, v_w, m_b, v_b, beta1, beta2, epsilon, lr, t):
    """
    Perform one update step of Adam optimization.
    
    Args:
        w, b: current parameters
        dw, db: gradients
        m_w, v_w, m_b, v_b: Adam momentum and velocity terms
        beta1, beta2: exponential decay rates
        epsilon: small constant for numerical stability
        lr: learning rate
        t: time step
        
    Returns:
        Updated parameters and Adam variables
    """
    t += 1
    lr_t = lr * (np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))

    # Update for weights
    m_w = beta1 * m_w + (1 - beta1) * dw
    v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
    m_hat_w = m_w / (1 - beta1 ** t)
    v_hat_w = v_w / (1 - beta2 ** t)
    w -= lr_t * m_hat_w / (np.sqrt(v_hat_w) + epsilon)

    # Update for bias
    m_b = beta1 * m_b + (1 - beta1) * db
    v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
    m_hat_b = m_b / (1 - beta1 ** t)
    v_hat_b = v_b / (1 - beta2 ** t)
    b -= lr_t * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    return w, b, m_w, v_w, m_b, v_b, t

def adam_optimization(X, y, w, b, alpha=0.01, num_iters=1000, batch_size=128, 
                      validation_data=None, verbose=True):
    """
    Implement Adam optimization algorithm.
    
    Args:
        X: training data, shape (m, n)
        y: labels, shape (m, classes)
        w: initial weights 
        b: initial bias
        alpha: learning rate
        num_iters: number of iterations
        batch_size: mini-batch size
        validation_data: tuple (X_val, y_val) for validation
        verbose: whether to print progress
        
    Returns:
        w, b: optimized parameters
        J_history: training cost history
        val_history: validation cost history (if validation_data provided)
    """
    m_w, v_w, m_b, v_b, beta1, beta2, epsilon, lr, t = initialize_adam(w, b, lr=alpha)
    J_history = []
    val_history = []
    
    # Unpack validation data if provided
    if validation_data is not None:
        X_val, y_val = validation_data
        has_validation = True
    else:
        has_validation = False

    # Training loop
    for i in range(num_iters):
        indices = np.random.permutation(X.shape[0])
        X_shuffled, y_shuffled = X[indices], y[indices]

        # Mini-batch gradient descent
        for j in range(0, X.shape[0], batch_size):
            # Get current batch
            end = min(j + batch_size, X.shape[0])
            X_batch = X_shuffled[j:end]
            y_batch = y_shuffled[j:end]
            
            # Compute gradients and update parameters
            dw, db = compute_gradient(X_batch, y_batch, w, b)
            w, b, m_w, v_w, m_b, v_b, t = adam_update(
                w, b, dw, db, m_w, v_w, m_b, v_b, beta1, beta2, epsilon, lr, t
            )

        # Compute cost on full training set and log
        cost = compute_cost(X, y, w, b)
        J_history.append(cost)
        
        # Compute validation cost if validation data provided
        if has_validation:
            val_cost = compute_cost(X_val, y_val, w, b)
            val_history.append(val_cost)
        
        # Print progress at intervals
        if verbose and (i % (num_iters // 10) == 0 or i == num_iters - 1):
            # Calculate training accuracy
            y_pred = softmax(np.dot(X, w) + b)
            train_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
            
            if has_validation:
                # Calculate validation accuracy
                y_val_pred = softmax(np.dot(X_val, w) + b)
                val_acc = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
                print(f"Iteration {i:4}: Cost {cost:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Iteration {i:4}: Cost {cost:.4f}, Train Acc: {train_acc:.4f}")

    return w, b, J_history, val_history if has_validation else None
