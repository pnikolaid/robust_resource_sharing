from parameters import test_scenario

if test_scenario == 1:
    import g_plot_original
elif test_scenario == 4:
    import g_plot_contrast
else:
    import g_plot_alternative