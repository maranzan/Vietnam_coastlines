import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime

sitename = 'Hoi_An_IA'
filepath = os.path.join(os.getcwd(), 'data', sitename, f'{sitename}_output.pkl')

with open(filepath, 'rb') as f:
    output = pickle.load(f)

dates = output['dates']
shorelines = output['shorelines']

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
ax_map = fig.add_subplot(gs[0, :])
ax_plot = fig.add_subplot(gs[1, :])
plt.subplots_adjust(bottom=0.2)

ref_idx = -1
sl_ref = shorelines[ref_idx]
ax_map.plot(sl_ref[:, 0], sl_ref[:, 1], 'k--', lw=1, label=f'RReference ({dates[ref_idx].date()})')
line_comp, = ax_map.plot(shorelines[0][:, 0], shorelines[0][:, 1], '.', lw=2, label='Comparison')

ax_map.set_title("Spatial comparison of Shorelines")
ax_map.axis('equal')
ax_map.legend()


#SLider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Date', 0, len(dates)-1, valinit=0, valfmt='%d')

def update(val):
    idx = int(slider.val)
    sl = shorelines[idx]
    line_comp.set_data(sl[:, 0], sl[:, 1])
    ax_map.set_title(f"Comparison : {dates[idx].date()} vs {dates[ref_idx].date()}")
    fig.canvas.draw_idle()

slider.on_changed(update)


def on_click(event):
    if event.inaxes != ax_map: return
    
    x_click, y_click = event.xdata, event.ydata
    
    distances_over_time = []
    for sl in shorelines:
        dist = np.sqrt((sl[:, 0] - x_click)**2 + (sl[:, 1] - y_click)**2)
        distances_over_time.append(np.min(dist))
    
    ax_plot.clear()
    ax_plot.plot(dates, distances_over_time, 'o-', color='teal')
    ax_plot.set_ylabel("Distance to point (m)")
    ax_plot.set_title(f"Temporal evolution at point [{int(x_click)}, {int(y_click)}]")
    ax_plot.grid(True, alpha=0.3)
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()