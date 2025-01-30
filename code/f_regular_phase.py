from parameters import test_scenario

if test_scenario == 1:
    import f_regular_phase_original
elif test_scenario == 4:
    import f_regular_phase_contrast
else:
    import f_regular_phase_alternative
