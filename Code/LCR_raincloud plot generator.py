"""
This file makes from the nodule_meta file per annotation a raincloud plot with malignancy
"""


import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import ptitprince as pt

## Variables
sns.set(style="whitegrid",font_scale=2)
savefigs = True
figs_dir = r'D:\Docs\BEP\LONG DATA\new_data\figs'
path_to_nodule_meta = r"D:\Docs\BEP\LONG DATA\new_data\new_nodule_info.xlsx"

# ---------
# Main code
# ---------

if savefigs:
    # Make the figures folder if it doesn't yet exist
    if not os.path.isdir(r'D:\Docs\BEP\LONG DATA\new_data\figs'):
        os.makedirs(r'D:\Docs\BEP\LONG DATA\new_data\figs')
        
def export_fig(axis,text, fname):
    if savefigs:
        axis.text()
        axis.savefig(fname, bbox_inches='tight')

df = pd.read_excel(path_to_nodule_meta, engine = 'openpyxl')
df = df.drop(df[df.Sickness == 'Inconclusive'].index)

dx = "Sickness"; dy = "subtlety"; ort = "h"; pal = "Set2"; sigma = .2
f, ax = plt.subplots(figsize=(7, 5))
pt.RainCloud(x = dx, y = dy, data = df, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort, move = .2)

plt.ylabel('Diagnostic label')

plt.title(f"Raincloud plot {dy} vs diagnostic label")
if savefigs:
    plt.savefig(os.path.join(figs_dir, f'raincloud_{dy}.png'), bbox_inches='tight')
    