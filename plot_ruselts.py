import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import seaborn as sns
import palettable
from matplotlib.pyplot import MultipleLocator
import numpy as np

def plot_fig3():

    loaded_dict = torch.load('/data/fig3_data.pt')

    # Reconstruct the DataFrame
    numeric_df = pd.DataFrame(loaded_dict['numeric_data'].numpy(), columns=loaded_dict['numeric_columns'])

    if loaded_dict['object_data']:
        object_df = pd.DataFrame(loaded_dict['object_data'], columns=loaded_dict['object_columns'])
        # Combine numeric and non-numeric data
        fig2_a_df = pd.concat([numeric_df, object_df], axis=1)
    else:
        fig2_a_df = numeric_df

    Bioactive_ligands = fig2_a_df[fig2_a_df['label'] == 'Bioactive ligands']
    FLAG = fig2_a_df[fig2_a_df['label'] == 'FLAG']
    Pocket2Mol = fig2_a_df[fig2_a_df['label'] == 'Pocket2Mol']
    AMDiff = fig2_a_df[fig2_a_df['label'] == 'AMDiff']

    box_name = ['Bioactive ligands', 'Pocket2Mol', 'FLAG', 'AMDiff']
    color_l = palettable.scientific.diverging.Roma_5.mpl_colors
    palette = [color_l[0], color_l[1], color_l[3], color_l[4]]

    result_path = '/results/r_fig3'
    os.makedirs(result_path, exist_ok=True)

    plot_dock, plot_logp, plot_qed, plot_sa = True, True, True, True,
    if plot_dock==True:
        # plot distribution of docking score
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)

        all_result_filter = pd.concat([Bioactive_ligands,  Pocket2Mol, FLAG, AMDiff])

        sns.kdeplot(Bioactive_ligands['dock score'], color=color_l[0], fill=True, label='Bioactive ligands', linewidth=2)
        sns.kdeplot(Pocket2Mol['dock score'], color=color_l[1], fill=True, label='Pocket2Mol', linewidth=2)
        sns.kdeplot(FLAG['dock score'], color=color_l[3], fill=True, label='FLAG', linewidth=2)
        sns.kdeplot(AMDiff['dock score'], color=color_l[4], fill=True, label='AMDiff', linewidth=2)

        sns.despine()
        plt.xlabel('Docking score (kcal/mol)', fontsize=28)
        plt.ylabel('Density', fontsize=28)

        plt.tick_params(labelsize = 26, axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', length=0)
        x_major_locator = MultipleLocator(5)
        y_major_locator = MultipleLocator(0.2)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlim(-14, -1)
        plt.ylim(0, 0.6)

        plt.tight_layout()
        plt.savefig(f'{result_path}/figs_b_1.jpg', dpi=600)

        plt.clf()
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)
        sns.boxplot(data=fig2_a_df, x="dock score", y="label", palette=palette, order=box_name, width=0.5, flierprops={"marker": "x"}, notch=True, showcaps=False, medianprops={"color": "coral"}, linewidth=1)

        sns.despine()
        plt.xlabel('Docking score (kcal/mol)', fontsize=28)
        plt.ylabel('Density', fontsize=28)
        plt.tick_params(labelsize = 26, axis="both", which="both", bottom="off", top="off", labelbottom="on",
                        left="off", right="off",
                        labelleft="on")
        # plt.tick_params(axis='y', labelrotation=45)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.get_yaxis().set_visible(False)

        ax.get_xaxis().tick_bottom()
        # ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', length=0)
        x_major_locator = MultipleLocator(5)
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-13, -1)
        plt.tight_layout()
        plt.savefig(f'{result_path}/figs_b_2.jpg', dpi=600)

    # plot distribution of logp score
    if plot_logp==True:
        plt.clf()
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)

        sns.kdeplot(Bioactive_ligands['logp score'], color=color_l[0], shade=True, label='Bioactive ligands', linewidth=2)
        sns.kdeplot(Pocket2Mol['logp score'], color=color_l[1], shade=True, label='Pocket2Mol', linewidth=2)
        sns.kdeplot(FLAG['logp score'], color=color_l[3], shade=True, label='FLAG', linewidth=2)
        sns.kdeplot(AMDiff['logp score'], color=color_l[4], shade=True, label='AMDiff', linewidth=2)
        sns.despine()
        plt.xlabel('LogP', fontsize=28)
        plt.ylabel('Density', fontsize=28)
        plt.legend(fontsize=28)
        plt.tick_params(labelsize=20, axis="both", which="both", bottom="off", top="off",
                        labelbottom="on", left="off", right="off", labelleft="on")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', length=0)
        x_major_locator = MultipleLocator(2)
        y_major_locator = MultipleLocator(0.3)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlim(-4.5, 6)
        plt.ylim(0, 0.65)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(0, 1), fontsize=22)

        plt.tight_layout()
        plt.savefig(f'{result_path}/figs_a_1.jpg', dpi=600)

        plt.clf()
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)
        sns.boxplot(data=fig2_a_df, x="logp score", y="label", order=box_name, palette=palette,
                    width=0.4, flierprops={"marker": "x"}, notch=True, showcaps=False,
                    medianprops={"color": "coral"}, linewidth=1.5)
        sns.despine()
        plt.xlabel('LogP', fontsize=28)
        plt.ylabel('Density', fontsize=28)
        plt.tick_params(labelsize=24, axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off",
                        right="off",
                        labelleft="on")
        plt.tick_params(axis='y', labelrotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', length=0)
        x_major_locator = MultipleLocator(2)
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-3, 5)
        # plt.xlim([-13, -5])
        plt.tight_layout()
        plt.savefig(f'{result_path}/figs_a_2.jpg', dpi=600)

    # plot distribution of qed score
    if plot_qed==True:
        plt.clf()
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)

        all_result_filter = pd.concat([Bioactive_ligands, Pocket2Mol, FLAG, AMDiff])

        sns.kdeplot(Bioactive_ligands['qed score'], color=color_l[0], shade=True, label='Bioactive ligands', linewidth=2)
        sns.kdeplot(Pocket2Mol['qed score'], color=color_l[1], shade=True, label='Pocket2Mol', linewidth=2)
        sns.kdeplot(FLAG['qed score'], color=color_l[3], shade=True, label='FLAG', linewidth=2)
        sns.kdeplot(AMDiff['qed score'], color=color_l[4], shade=True, label='AMDiff', linewidth=2)
        sns.despine()
        plt.xlabel('QED', fontsize=28)
        plt.ylabel('Density', fontsize=28)
        # plt.legend(fontsize=18, loc=1)
        plt.tick_params(labelsize=26, axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off",
                        right="off", labelleft="on")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', length=0)
        x_major_locator = MultipleLocator(0.4)
        y_major_locator = MultipleLocator(2)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlim(-0.15, 1.2)
        plt.ylim(0, 4.5)
        plt.tight_layout()
        plt.savefig(f'{result_path}/figs_c_1.jpg', dpi=600)

        plt.clf()
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)
        sns.boxplot(data=fig2_a_df, x="qed score", y="label", order=box_name, palette=palette,
                    width=0.4, flierprops={"marker": "x"}, notch=True, showcaps=False,
                    medianprops={"color": "coral"}, linewidth=1.5)
        sns.despine()
        plt.xlabel('QED', fontsize=28)
        plt.ylabel('Density', fontsize=28)
        plt.tick_params(labelsize=26, axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off",
                        right="off",
                        labelleft="on")
        # plt.tick_params(axis='y', labelrotation=45)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.get_yaxis().set_visible(False)

        ax.get_xaxis().tick_bottom()
        # ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', length=0)
        x_major_locator = MultipleLocator(0.4)
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f'{result_path}/figs_c_2.jpg', dpi=600)


    # plot distribution of sa score
    if plot_sa==True:
        plt.clf()
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)

        sns.kdeplot(Bioactive_ligands['sa score'], color=color_l[0], shade=True, label='Bioactive ligands', linewidth=2)
        sns.kdeplot(Pocket2Mol['sa score'], color=color_l[1], shade=True, label='Pocket2Mol', linewidth=2)
        sns.kdeplot(FLAG['sa score'], color=color_l[3], shade=True, label='FLAG', linewidth=2)
        sns.kdeplot(AMDiff['sa score'], color=color_l[4], shade=True, label='AMDiff', linewidth=2)
        sns.despine()
        plt.xlabel('SA', fontsize=28)
        plt.ylabel('Density', fontsize=28)
        # plt.legend(fontsize=18)
        plt.tick_params(labelsize=26, axis="both", which="both", bottom="off", top="off",
                        labelbottom="on", left="off", right="off", labelleft="on")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', length=0)
        x_major_locator = MultipleLocator(0.4)
        y_major_locator = MultipleLocator(4)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlim(-0.15, 1.1)
        plt.ylim(0, 9)
        plt.tight_layout()
        plt.savefig(f'{result_path}/figs_d_1.jpg', dpi=600)

        plt.clf()
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)
        sns.boxplot(data=fig2_a_df, x="sa score", y="label", order=box_name, palette=palette,
                    width=0.4, flierprops={"marker": "x"}, notch=True, showcaps=False,
                    medianprops={"color": "coral"}, linewidth=1.5)
        sns.despine()
        plt.xlabel('SA', fontsize=28)
        plt.ylabel('Density', fontsize=28)
        plt.tick_params(labelsize=26, axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off",
                        right="off",
                        labelleft="on")
        # plt.tick_params(axis='y', labelrotation=45)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.get_yaxis().set_visible(False)

        ax.get_xaxis().tick_bottom()
        # ax.get_yaxis().tick_left()

        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', length=0)

        x_major_locator = MultipleLocator(0.4)
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(0, 1)
        # plt.xlim([-13, -5])
        plt.tight_layout()
        plt.savefig(f'{result_path}/figs_d_2.jpg', dpi=600)

if __name__ == "__main__":

    plot_fig3()