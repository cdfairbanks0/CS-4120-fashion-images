import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow

def logConfusionMatrix(yTrue, yPred, classNames, title, filename="plot3 Confusion Matrix.png"):
    mlflow.start_run()
    cm = confusion_matrix(yTrue, yPred)
    fig, axes = plt.subplots()
    plot = ConfusionMatrixDisplay(cm, display_labels=classNames)
    plot.plot(ax=axes, xticks_rotation=45, colorbar=False)
    axes.set_title(title)
    fig.tight_layout()
    mlflow.log_figure(fig, filename)
    plt.close(fig)