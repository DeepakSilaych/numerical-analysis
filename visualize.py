import numpy as np
import matplotlib.pyplot as plt

def plot_loss(J_history, val_history=None):
    """
    Plot the loss function over iterations.
    
    Args:
        J_history: training loss history
        val_history: optional validation loss history
    """
    plt.figure(figsize=(10, 5))
    plt.plot(J_history, label='Training Loss')
    
    if val_history is not None:
        plt.plot(val_history, label='Validation Loss')
        plt.legend()
    
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Function Over Iterations")
    plt.grid(True)
    plt.show()

def plot_sample_predictions(X, y_true, y_pred, n_samples=5):
    """
    Display sample images with their predicted labels.
    
    Args:
        X: input images (flattened)
        y_true: true labels (one-hot encoded)
        y_pred: predicted probabilities
        n_samples: number of samples to display
    """
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples*2, 2))
    
    if n_samples == 1:
        axes = [axes]
    
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img = X[idx].reshape(28, 28)
        true_label = np.argmax(y_true[idx])
        pred_label = np.argmax(y_pred[idx])
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """
    Create and plot confusion matrix.
    
    Args:
        y_true: true labels (one-hot encoded)
        y_pred: predicted probabilities
    """
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    cm = np.zeros((10, 10), dtype=int)
    for i in range(len(y_true)):
        cm[y_true_labels[i]][y_pred_labels[i]] += 1
    
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

def plot_accuracy_comparison(iterations, accuracies, labels=None, title="Accuracy Comparison"):
    """
    Plot accuracy comparison between different models or configurations.
    
    Args:
        iterations: x-axis values (typically iteration numbers)
        accuracies: list of accuracy histories to compare
        labels: labels for each accuracy history
        title: plot title
    """
    plt.figure(figsize=(10, 6))
    
    for i, acc in enumerate(accuracies):
        if labels and i < len(labels):
            plt.plot(iterations, acc, label=labels[i])
        else:
            plt.plot(iterations, acc, label=f"Model {i+1}")
    
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show() 