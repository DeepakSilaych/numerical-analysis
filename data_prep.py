import numpy as np

def load_mnist_from_csv(train_path, test_path, normalize=True):
    """
    Load MNIST dataset from CSV files.
    
    Args:
        train_path: path to training CSV file
        test_path: path to test CSV file
        normalize: whether to normalize pixel values to [0, 1]
        
    Returns:
        (x_train, y_train), (x_test, y_test): training and test data/labels
    """
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

def one_hot_encode(labels, num_classes=10):
    """
    Convert class labels to one-hot encoded vectors.
    
    Args:
        labels: array of integer class labels
        num_classes: number of classes
        
    Returns:
        one-hot encoded vectors
    """
    return np.eye(num_classes)[labels]

def train_validation_split(X, y, val_size=0.1, random_state=None):
    """
    Split the dataset into training and validation sets.
    
    Args:
        X: input data
        y: target labels
        val_size: proportion of data to use for validation
        random_state: random seed for reproducibility
        
    Returns:
        X_train, X_val, y_train, y_val: split data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    val_samples = int(n_samples * val_size)
    
    val_indices = indices[:val_samples]
    train_indices = indices[val_samples:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    return X_train, X_val, y_train, y_val

def preprocess_data(train_path, test_path, val_size=0.1, random_state=42):
    """
    Load and preprocess MNIST data.
    
    Args:
        train_path: path to training CSV file
        test_path: path to test CSV file
        val_size: proportion of training data to use for validation
        random_state: random seed for reproducibility
        
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test): processed data
    """
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_from_csv(train_path, test_path)
    
    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_validation_split(
        x_train, y_train, val_size=val_size, random_state=random_state
    )
    
    # One-hot encode labels
    y_train = one_hot_encode(y_train)
    y_val = one_hot_encode(y_val)
    y_test = one_hot_encode(y_test)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test) 