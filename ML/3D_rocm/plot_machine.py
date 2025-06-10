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

    l2_history = np.load('l2.npy')
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="L2 error", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("L2")
    plt.yscale("log")  # Use log scale if losses vary widely
    plt.legend()
    plt.grid(True)
    plt.title("Deep Ritz Method Training L2 Error")
    plt.savefig("l2_plot.png")  # Save the plot
    plt.show()

    h1_history = np.load('h1.npy')
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="H1 error", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("H1")
    plt.yscale("log")  # Use log scale if losses vary widely
    plt.legend()
    plt.grid(True)
    plt.title("Deep Ritz Method Training H1 Error")
    plt.savefig("h1_plot.png")  # Save the plot
    plt.show()
