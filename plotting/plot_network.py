import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])

def plot_network(network):
    plt.figure(figsize = (2,2))
    for node in network:
        if node.index == 0:
            plt.scatter(node.x, node.y,
                    c = "red", edgecolors = "black",
                    s = 6, linewidths=0.5)
        else:
            plt.scatter(node.x, node.y,
                    c = "blue", edgecolors = "black",
                    s = 4, linewidths=0.5)
        plt.text(node.x + 2, node.y + 2,
                 f"Node {node.index}", fontsize = 4)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid(True)
    plt.show()