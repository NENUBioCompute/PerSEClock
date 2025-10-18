import matplotlib.pyplot as plt

def plot_predicted_vs_true(pred, true, title="Predicted vs True Age", save_path=None):
    """
    Plot the relationship between predicted and true ages.

    Parameters
    ----------
    pred : list or np.ndarray
        Predicted ages.
    true : list or np.ndarray
        True ages.
    title : str
        Plot title.
    save_path : str or None
        Optional path to save the figure.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.6, edgecolors="k")
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--', label="Ideal Fit")
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
