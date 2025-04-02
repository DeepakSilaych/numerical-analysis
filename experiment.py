import numpy as np
import time
import matplotlib.pyplot as plt
from data_prep import preprocess_data
from mnist_model import initialize_weights, evaluate_model
from optimizers import adam_optimization
from visualize import plot_accuracy_comparison

def run_experiment(x_train, y_train, x_val, y_val, x_test, y_test, configs):
    """
    Run experiments with different hyperparameter configurations.
    
    Args:
        x_train, y_train: training data
        x_val, y_val: validation data
        x_test, y_test: test data
        configs: list of dictionaries with hyperparameter configurations
        
    Returns:
        results: dictionary with experiment results
    """
    results = {
        'configs': configs,
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'training_time': [],
        'loss_history': [],
        'val_history': [],
    }
    
    for i, config in enumerate(configs):
        print(f"\n{'-'*50}")
        print(f"Running experiment {i+1}/{len(configs)}")
        print(f"Configuration: {config}")
        print(f"{'-'*50}")
        
        # Initialize model parameters
        input_size = x_train.shape[1]
        output_size = 10
        w, b = initialize_weights(input_size, output_size, method=config.get('init_method', 'random'))
        
        # Start timing
        start_time = time.time()
        
        # Train model with current configuration
        w, b, J_history, val_history = adam_optimization(
            x_train, y_train, w, b,
            alpha=config['alpha'],
            num_iters=config['num_iters'],
            batch_size=config['batch_size'],
            validation_data=(x_val, y_val),
            verbose=config.get('verbose', True)
        )
        
        # End timing
        training_time = time.time() - start_time
        
        # Evaluate model
        train_acc, _ = evaluate_model(x_train, y_train, w, b)
        val_acc, _ = evaluate_model(x_val, y_val, w, b)
        test_acc, _ = evaluate_model(x_test, y_test, w, b)
        
        # Store results
        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)
        results['test_acc'].append(test_acc)
        results['training_time'].append(training_time)
        results['loss_history'].append(J_history)
        results['val_history'].append(val_history)
        
        # Print results
        print(f"\nResults for experiment {i+1}:")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
    
    return results

def plot_experiment_results(results):
    """
    Plot experiment results.
    
    Args:
        results: dictionary with experiment results
    """
    configs = results['configs']
    
    # Create labels for configurations
    labels = [f"lr={c['alpha']}, bs={c['batch_size']}" for c in configs]
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    x = np.arange(len(configs))
    width = 0.25
    
    plt.bar(x - width, results['train_acc'], width, label='Train')
    plt.bar(x, results['val_acc'], width, label='Validation')
    plt.bar(x + width, results['test_acc'], width, label='Test')
    
    plt.xlabel('Configuration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training time
    plt.subplot(1, 2, 2)
    plt.bar(x, results['training_time'])
    plt.xlabel('Configuration')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for i, J_history in enumerate(results['loss_history']):
        plt.plot(J_history, label=labels[i])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i, val_history in enumerate(results['val_history']):
        if val_history is not None:
            plt.plot(val_history, label=labels[i])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Define paths to data files
    train_path = 'data/mnist_train.csv'
    test_path = 'data/mnist_test.csv'
    
    # Preprocess data
    print("Preprocessing data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(
        train_path, test_path, val_size=0.1, random_state=42
    )
    
    # Define configurations to test
    configs = [
        {
            'alpha': 0.001,
            'batch_size': 128,
            'num_iters': 500,
            'init_method': 'random',
            'verbose': False
        },
        {
            'alpha': 0.01,
            'batch_size': 128,
            'num_iters': 500,
            'init_method': 'random',
            'verbose': False
        },
        {
            'alpha': 0.05,
            'batch_size': 128,
            'num_iters': 500,
            'init_method': 'random',
            'verbose': False
        },
        {
            'alpha': 0.05,
            'batch_size': 256,
            'num_iters': 500,
            'init_method': 'xavier',
            'verbose': False
        }
    ]
    
    # Run experiments
    print("\n" + "="*50)
    print("Running MNIST experiments with Adam optimizer")
    print("="*50)
    
    results = run_experiment(x_train, y_train, x_val, y_val, x_test, y_test, configs)
    
    # Plot results
    print("\nPlotting experiment results...")
    plot_experiment_results(results)
    
    # Find best configuration
    best_idx = np.argmax(results['test_acc'])
    best_config = configs[best_idx]
    
    print("\n" + "="*50)
    print("Best Configuration:")
    print("="*50)
    print(f"Learning rate: {best_config['alpha']}")
    print(f"Batch size: {best_config['batch_size']}")
    print(f"Init method: {best_config['init_method']}")
    print(f"Training accuracy: {results['train_acc'][best_idx]:.4f}")
    print(f"Validation accuracy: {results['val_acc'][best_idx]:.4f}")
    print(f"Test accuracy: {results['test_acc'][best_idx]:.4f}")
    print(f"Training time: {results['training_time'][best_idx]:.2f} seconds")
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main() 