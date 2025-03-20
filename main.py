# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# Main program
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# local functions
import enso_ocean_dynamics.script_figures as fig
# ---------------------------------------------------#


# collect the figure scripts
figure_scripts = {
    "f1": fig.f01_time_series_plot,
}
figure_calling_names = ", ".join(figure_scripts.keys())


if __name__ == '__main__':
    figure_number = input("Which figure do you want to plot?\n     Please enter one of: %s\n" % figure_calling_names)
    while figure_number not in list(figure_scripts.keys()):
        figure_number = input("Given value %s does not correspond to a figure\n     Please enter one of: %s\n" % (
            figure_number, figure_calling_names))
    figure_scripts[figure_number]()