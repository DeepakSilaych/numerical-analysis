import numpy as np
import time
from data_prep import preprocess_data
from mnist_model import initialize_weights, evaluate_model
from optimizers import adam_optimization
from visualize import plot_loss, plot_sample_predictions, plot_confusion_matrix

def main():
    # Define paths to data files
    train_path = 'data/mnist_train.csv'
    test_path = 'data/mnist_test.csv'
    
    # Preprocess data
    print("Preprocessing data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(
        train_path, test_path, val_size=0.1, random_state=42
    )
    
    # Print dataset shapes
    print(f"\nDataset shapes:")
    print(f"Training:   {x_train.shape[0]} samples")
    print(f"Validation: {x_val.shape[0]} samples")
    print(f"Test:       {x_test.shape[0]} samples")
    
    # Initialize model parameters
    print("\nInitializing model parameters...")
    input_size = x_train.shape[1]  # 784 (28x28)
    output_size = 10  # 10 classes (digits 0-9)
    w, b = initialize_weights(input_size, output_size, method='random')
    
    # Define training hyperparameters
    hyperparams = {
        'alpha': 0.05,      # Learning rate
        'num_iters': 1000,  # Number of iterations
        'batch_size': 128,  # Batch size
    }
    
    # Train the model with Adam optimizer
    print("\n" + "="*50)
    print("Training with Adam optimizer:")
    print("="*50)
    print(f"Learning rate: {hyperparams['alpha']}")
    print(f"Batch size: {hyperparams['batch_size']}")
    print(f"Number of iterations: {hyperparams['num_iters']}")
    
    # Start timing
    start_time = time.time()
    
    # Train model
    w, b, J_history, val_history = adam_optimization(
        x_train, y_train, w, b,
        alpha=hyperparams['alpha'],
        num_iters=hyperparams['num_iters'],
        batch_size=hyperparams['batch_size'],
        validation_data=(x_val, y_val),
        verbose=True
    )
    
    # End timing
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate model
    train_acc, _ = evaluate_model(x_train, y_train, w, b)
    val_acc, _ = evaluate_model(x_val, y_val, w, b)
    test_acc, y_pred = evaluate_model(x_test, y_test, w, b)
    
    # Print results
    print("\nAdam Optimizer Results:")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot loss curve
    print("\nPlotting loss curve...")
    plot_loss(J_history, val_history)
    
    # Plot sample predictions
    print("\nPlotting sample predictions...")
    plot_sample_predictions(x_test, y_test, y_pred, n_samples=5)
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main() 