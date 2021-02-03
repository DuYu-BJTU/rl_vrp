import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_5(small, mid, large):
    fig2 = plt.figure(figsize=(40, 40), dpi=500, constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
    f2_ax1 = fig2.add_subplot(spec2[0, 0])
    f2_ax2 = fig2.add_subplot(spec2[0, 1])
    f2_ax3 = fig2.add_subplot(spec2[1, 0])
    plt.savefig("figure/test.png")
    plt.close()


if __name__ == '__main__':
    plot_5(0, 0, 0)
