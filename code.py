import numpy as np
import matplotlib.pyplot as plt

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_samples = int(n_samples * test_size)
    
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def load_mnist_from_csv(train_path, test_path, normalize=True):
    print("Loading training data...")
    train_data = np.loadtxt(train_path, delimiter=',')
    print("Loading test data...")
    test_data = np.loadtxt(test_path, delimiter=',')
    
    y_train = train_data[:, 0].astype(np.int32)
    x_train = train_data[:, 1:]
    
    y_test = test_data[:, 0].astype(np.int32)
    x_test = test_data[:, 1:]
    
    if normalize:
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    
    print(f"Loaded {x_train.shape[0]} training samples and {x_test.shape[0]} test samples")
    return (x_train, y_train), (x_test, y_test)

print("Loading MNIST dataset from CSV files...")
(x_train, y_train), (x_test, y_test) = load_mnist_from_csv(
    '/home/deepak-silaych/Desktop/sem6/numerical/project/data/mnist_train.csv',
    '/home/deepak-silaych/Desktop/sem6/numerical/project/data/mnist_test.csv'
)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train = one_hot_encode(y_train)
y_val = one_hot_encode(y_val)
y_test = one_hot_encode(y_test)

w = np.random.randn(x_train.shape[1], 10) * 0.01
b = np.zeros(10)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_cost(X, y, w, b, lambda_=1):
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = softmax(z)
    cross_entropy_cost = -np.sum(y * np.log(f_wb + 1e-8)) / m
    
    regularization_cost = 0
    if lambda_ > 0:
        regularization_cost = (lambda_ / (2 * m)) * np.sum(np.square(w))
    
    return cross_entropy_cost + regularization_cost

def compute_gradient(X, y, w, b, lambda_=0):
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = softmax(z)
    error = f_wb - y
    
    dj_dw = np.dot(X.T, error) / m
    
    if lambda_ > 0:
        dj_dw += (lambda_ / m) * w
        
    dj_db = np.sum(error, axis=0) / m
    return dj_db, dj_dw

def gradient_descent(X, y, X_val, y_val, w_in, b_in, cost_function, gradient_function, 
                    alpha, num_iters, batch_size, lambda_, patience=5, min_delta=0.001,
                    decay_rate=0.95, decay_steps=100):
    m = X.shape[0]
    J_history = []
    val_history = []
    w_history = []
    
    best_val_cost = float('inf')
    best_w = None
    best_b = None
    patience_counter = 0
    
    for i in range(num_iters):
        current_alpha = alpha * (decay_rate ** (i // decay_steps))
        
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]
            
            dj_db, dj_dw = gradient_function(X_batch, y_batch, w_in, b_in, lambda_)
            w_in -= current_alpha * dj_dw
            b_in -= current_alpha * dj_db
        
        cost = cost_function(X, y, w_in, b_in, lambda_)
        J_history.append(cost)
        
        val_cost = cost_function(X_val, y_val, w_in, b_in, lambda_)
        val_history.append(val_cost)
        
        if i % (num_iters // 10) == 0 or i == num_iters - 1:
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {cost:.4f}, Val Cost {val_cost:.4f}, Alpha {current_alpha:.6f}")
        
        if val_cost < best_val_cost - min_delta:
            best_val_cost = val_cost
            best_w = w_in.copy()
            best_b = b_in.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at iteration {i}")
            return best_w, best_b, J_history, val_history, w_history
    
    if best_w is not None:
        return best_w, best_b, J_history, val_history, w_history
    else:
        return w_in, b_in, J_history, val_history, w_history

alpha = 0.1
num_iters = 2000
batch_size = 256
lambda_ = 0.01

w, b, J_history, val_history, w_history = gradient_descent(
    x_train, y_train, x_val, y_val, w, b, compute_cost, compute_gradient, 
    alpha, num_iters, batch_size, lambda_, patience=10
)

print("Training complete!")

def plot_loss(J_history, val_history):
    plt.figure(figsize=(10, 6))
    plt.plot(J_history, label='Training Loss')
    plt.plot(val_history, label='Validation Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Function Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(J_history, val_history)

def evaluate_model(X, y, w, b):
    z = np.dot(X, w) + b
    y_pred = softmax(z)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
    return accuracy, y_pred

val_acc, val_pred = evaluate_model(x_val, y_val, w, b)
test_acc, y_pred = evaluate_model(x_test, y_test, w, b)

print(f'Validation accuracy: {val_acc:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

def plot_confusion_matrix(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred)

def plot_sample_predictions(X, y_true, y_pred, n_samples=5):
    fig, axes = plt.subplots(1, n_samples, figsize=(12, 3))
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        img = X[idx].reshape(28, 28)
        true_label = np.argmax(y_true[idx])
        pred_label = np.argmax(y_pred[idx])
        
        ax.imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

plot_sample_predictions(x_test, y_test, y_pred, n_samples=8)