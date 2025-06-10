import matplotlib.pyplot as plt
import numpy as np

def plotter():
    loss_history = np.load('loss.npy')
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Total Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.yscale("log")  # Use log scale if losses vary widely
    plt.legend()
    plt.grid(True)
    plt.title("Deep Ritz Method Training Loss")
    plt.savefig("loss_plot.png")  # Save the plot
    plt.show()