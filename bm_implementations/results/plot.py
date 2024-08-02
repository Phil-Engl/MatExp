import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = os.path

csvfiles = sorted(glob.glob('*.csv'))

implementations = dict()

for file in csvfiles:
    label = file.split('.')[0]
    
    if(label.startswith("base")):
        split = label.split('_')
        label = split[0] + "_" + split[2]
    elif(label.startswith("runtime")):
        split = label.split('_')
        if(split[1] == "1"):
            label = "no unrolling_" + split[4]
        else:
            label = "unrolled_" + split[1] + "_times_" + split[4]  
    
    
    readfdf = pd.read_csv(file)
    impls = readfdf.Function.unique().tolist()
    
    for im in impls:
        impldf = readfdf[(readfdf['Function'] == im)]
        impldf = impldf.assign(benchmark = label)
        if(im in implementations):
            df = implementations.get(im)
            df = pd.concat([df, impldf])
            implementations[im] = df
        else:
            implementations[im] = impldf
      
for key in implementations.keys():
    fig = plt.figure()
    fig_ax = fig.gca()
    markers = ['o', 'v', 's', '^', '<', '>', 'p', 'h']
    colors = ["firebrick", "forestgreen", "rebeccapurple", "goldenrod", "lavender", "lightblue", "turquoise", "chocolate"]

    
    df = implementations[key]
    impls = df.benchmark.unique().tolist()
    mrk = 0
    for im in impls:
        extracted = df[(df['benchmark']) == im]
        x_values = np.array(list(extracted['n']))
        y_values = np.array(list(extracted['floats/cycle']))
        print(x_values)
        print(y_values)

        fig_ax.plot(
            x_values, 
            y_values, 
            marker=markers[mrk],
            markersize=4,
            label=im,
            color=colors[mrk]
        )
        mrk = mrk + 1


        fig_ax.set_xlabel("Input size ($n*n$)", fontsize=11)
        fig_ax.set_ylabel("Performance (flops/cycle)", fontsize=11, loc="top", rotation="horizontal", labelpad=-153)
        #handles, labels = fig_ax.get_legend_handles_labels()
        #order = [1,0]
        #fig_ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='right', fontsize=10, framealpha=1)
        fig_ax.legend()
        fig_ax.grid(True, color='white', axis="y", linestyle="--", linewidth=1)
        fig_ax.set_facecolor(color="gainsboro")
        fig_ax.spines["right"].set_visible(False)
        fig_ax.spines["top"].set_visible(False)
        fig_ax.set_xticks(range(0,1024,128))

        #fig_ax.axvline(x = 13, color="black", linewidth = 0.6, label = "L1")
        #fig_ax.text(8, 1.82, "L1", fontsize=10)
        #fig_ax.axvline(x = 15, color="black", linewidth = 0.6, label = "L2")
        #fig_ax.text(13.8, 1.82, "L2", fontsize=10)
        #fig_ax.axvline(x = 20, color="black", linewidth = 0.6, label = "L3")
        #fig_ax.text(17, 1.82, "L3", fontsize=10)
        #fig_ax.text(21.8, 1.82, "RAM", fontsize=10)


        #plt.tight_layout()
        plt.savefig(key + ".pdf")
    