import matplotlib.pyplot as plt


def init_plot_settings():
    plt.style.use("seaborn-v0_8-muted")

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.format"] = "pdf"
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.1

    plt.rcParams["figure.titlesize"] = 18
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 18

    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 16
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["axes.titlepad"] = 6

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["mathtext.it"] = "serif:italic"
    plt.rcParams["lines.marker"] = ""
    plt.rcParams["legend.frameon"] = False
