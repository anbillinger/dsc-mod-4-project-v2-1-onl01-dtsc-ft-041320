import numpy as np
import itertools
import matplotlib.pyplot as plt

def main(cm, classes,normalize=False,title='Confusion matrix',cmap=None):

	# Check if normalize is set to True
	# If so, normalize the raw confusion matrix before visualizing
	if normalize: cm = np.round(cm/sum(sum(cm))*100,2)
	print(cm)

	if cmap == None: cmap=plt.cm.Blues

	plt.imshow(cm, cmap=cmap)
	
	# Add title and axis labels 
	plt.title(title) 
	plt.ylabel('True label') 
	plt.xlabel('Predicted label')
	
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes)
	plt.yticks(tick_marks, classes)
	
	# Text formatting
	# Add labels to each cell
	thresh = cm.max() / 2.
	if normalize: cm_txt = np.core.defchararray.add(cm.astype('str'),'%')
	# Here we iterate through the confusion matrix and append labels to our visualization 
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if not normalize:
			plt.text(j, i, cm[i, j],horizontalalignment='center',color='white' if cm[i, j] > thresh else 'black')
		else: plt.text(j, i, cm_txt[i, j],horizontalalignment='center',color='white' if cm[i, j] > thresh else 'black')
	
	# Add a legend
	plt.colorbar()
	plt.show() 
