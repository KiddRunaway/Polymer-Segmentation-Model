import numpy as np
import scipy as sp
import functions as fun
import model_functions as mf
from matplotlib import cm
import matplotlib.pyplot as plt
import time

"Parameters"
#Number of grid cells for molar mass distribution
N_x = 200 

#Number of breakage events
N_t = 50

#Breakage location gaussian parameter (sigma)
r = 0.09

#Proportionality constant
s = 1.5

"Data Processing"
#Read data from t=5.45 min to 9.8 min and delete all NaN points
data = fun.delete_nan(fun.excel_reader("PS 20 Hz.xlsx", 5.45, 9.8))

#Retention time calibration function
mw_array, ret_time = {}, data.iloc[:, 0]
mw_array['A'], mw_array['B'] = np.log10(fun.calibration(ret_time, "PMMA")['A']), np.log10(fun.calibration(ret_time, "PMMA")['B'])
data = fun.ret_to_logmw_file(data, "PMMA")

#Normalize and convert intensity to fractions and interpolate such that data fits N_x
frac_data = fun.interpolator(fun.fractionater(data), mw_array, N_x)

"Calculations"
x_array = np.arange(1, N_x+1, 1)
t_array = np.arange(0, N_t+1, 1)

#Initial Condition
f_0 = frac_data["By1"]

start = time.time()
sol = mf.model(f_0, N_x, N_t, r, s)
end = time.time()
print("The whole solution took %.3f seconds to compute" % (end-start))
plt.plot(x_array, frac_data["By5"], linestyle="dashed", color="black", label="experimental")
plt.plot(x_array, sol[:, -1], color="blue", label="model")
plt.legend()
plt.show()

mf.contour_plotter(sol, x_array, t_array, 2)