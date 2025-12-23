# IGDM Time-Weighted RRT-Infotaxis implementation
# Time-dependent weights: w1(t) = 0.6 - 0.005*t, w2(t) = 1 - w1(t)
# Transitions from exploration (0.6*J1 - 0.4*J2) to exploitation (0.4*J1 - 0.6*J2)
