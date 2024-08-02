import matplotlib.pyplot as plt
import numpy as np

def format_matrix_names(matrices):
    return [(s.removeprefix('data/').removeprefix('small/').removeprefix('medium/')).removesuffix('.txt') for s in matrices]


def make_plot(matrices, impls, cycles, title, y_label):
    _, ax = plt.subplots()

    matrices = format_matrix_names(matrices)


    xticks = range(len(matrices))
    width = 0.9 / len(impls)

    for (idx, impl) in enumerate(impls):
        position = [xticks[i] + (width*(1-len(impls))/2) + idx*width for (i, _) in enumerate(matrices)]
        plt.bar(position, cycles[idx], width=width, label=impl)
        

    plt.ylabel(y_label, rotation=0)
    plt.xlabel("matrix", rotation=0)
    plt.grid(alpha=1, c='white', linewidth=0.5)
    plt.title(title, y=1.1, x=0.435)
    plt.legend(bbox_to_anchor=(1, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('gainsboro')
    plt.xticks(xticks, matrices, rotation = 90)
    plt.show()

def make_performance_plot(matrices, impls, perf):
    _, ax = plt.subplots()

    matrices = format_matrix_names(matrices)


    xticks = range(len(matrices))
    width = 0.9 / len(impls)

    for (idx, impl) in enumerate(impls):
        position = [xticks[i] + (width*(1-len(impls))/2) + idx*width for (i, _) in enumerate(matrices)]
        plt.bar(position, perf[idx], width=width, label=impl)
        

    plt.ylabel("performances [flops/cycle]", rotation=0)
    plt.xlabel("matrix", rotation=0)
    plt.grid(alpha=1, c='white', linewidth=0.5)
    plt.title('Performance results of matrix exponentials', y=1.1, x=0.435)
    plt.legend(bbox_to_anchor=(1, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('gainsboro')
    plt.xticks(xticks, matrices, rotation = 90)
    plt.show()

def make_all_plots(matrices, impls, cycles, perf, flops):
    make_plot(matrices, impls, cycles, "Runtime of matrix exponential", "runtime [cycles]")
    make_plot(matrices, impls, perf, "Performance of matrix exponential", "performance [flops/cycle]")
    make_plot(matrices, impls, flops, "Flop count of matrix exponential", "#flops")