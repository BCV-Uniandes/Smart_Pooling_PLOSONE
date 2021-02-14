import os
import pickle 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from matplotlib.font_manager import FontProperties
from sklearn import metrics



def plot_results(namefile,output_dir):
	# Open and read file data. Create variables to use in the plot section
	with open(namefile,'rb') as file:
		prevalence, vec_standardEff, vec_smartEff= pickle.load(file)

	vec_standardEff = 1/(np.array(vec_standardEff))
	vec_smartEff = 1/(np.array(vec_smartEff))
	One_efficiency_limit = np.ones(len(vec_smartEff))

	# TestCenterDataset plot -----------------------------------------------------------------------------------------------

	fig, ax = plt.subplots(1,1,figsize=(7,6))

	# Title
	ax.margins(0)
	ax.set_title('Metadata from test center',fontsize=12)

	smart, = ax.plot(prevalence, vec_smartEff, label='Smart pooling', marker='o', color='black')

	standard, = ax.plot(prevalence, vec_standardEff, label='Dorfman testing', linestyle='dotted', color='black')#, marker='o', markersize=4)
	Eff_limitOne, = ax.plot(prevalence, One_efficiency_limit, label='Individual testing' , linestyle='dashed' , color='black', alpha=0.4)

	# Solid fill
	fill = ax.fill_between(prevalence, One_efficiency_limit, One_efficiency_limit+0.2, label='No improvement', facecolor='red', alpha=0.3)

	# Axis labels
	ax.set_xlabel('Prevalence [%]')
	ax.set_ylabel('Expected number of tests per specimen')

	box = ax.get_position()

	fontP = FontProperties()
	fontP.set_size('small')

	# Legend inside the graph
	ax.legend(handles=[fill, smart, standard, Eff_limitOne], loc='lower right',fancybox=True, shadow=True, ncol=1,prop=fontP, frameon=False)

	ax.set(xlim=(5,25))
	ax.set(ylim=(0,1.1))

	plt.show()

	# pdb.set_trace()

	fig.savefig(os.path.join(output_dir,'TestCenterDataset_ExpectedNumberOfTests.pdf'), dpi=600) # Route and name of the output file containing the generated plot