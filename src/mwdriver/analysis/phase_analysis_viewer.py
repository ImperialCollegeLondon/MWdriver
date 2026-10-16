"""Interactive Matplotlib viewer for offline phase-analysis results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


def show_analysis_result_viewer(results):
    figure = plt.figure(figsize=(12, 8))
    figure.canvas.manager.set_window_title("Phase Analysis Result Viewer")
    image_axis = figure.add_axes([0.08, 0.08, 0.76, 0.72])
    colorbar_axis = figure.add_axes([0.87, 0.12, 0.025, 0.64])
    buttons = []

    def display_result(result_index):
        result = results[result_index]
        image_axis.clear()
        colorbar_axis.clear()
        plot_type = result.get("plot_type", "image")
        if plot_type == "bar":
            image_axis.bar(result["labels"], result["values"])
            image_axis.axhline(0, linewidth=1)
            image_axis.set_xlabel("Noll index")
            image_axis.set_ylabel(result["ylabel"])
        else:
            image = image_axis.imshow(
                result["image"],
                cmap=result["cmap"],
                vmin=result.get("vmin"),
                vmax=result.get("vmax"),
            )
            image_axis.set_xlabel("x pixel")
            image_axis.set_ylabel("y pixel")
            figure.colorbar(
                image,
                cax=colorbar_axis,
                label=result["colorbar"],
            )
        image_axis.set_title(result["title"])
        figure.canvas.draw_idle()

    buttons_per_row = 6
    button_width = 0.14
    button_height = 0.045
    button_x_start = 0.035
    button_x_gap = 0.155
    button_y_positions = [0.94, 0.89, 0.84]

    for index, result in enumerate(results):
        row = index // buttons_per_row
        column = index % buttons_per_row
        button_axis = figure.add_axes([
            button_x_start + column * button_x_gap,
            button_y_positions[row],
            button_width,
            button_height,
        ])
        button = Button(button_axis, result["button"])
        button.on_clicked(
            lambda event, result_index=index: display_result(result_index)
        )
        buttons.append(button)

    display_result(0)
    plt.show()