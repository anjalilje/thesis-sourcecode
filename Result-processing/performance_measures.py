## Load packages
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve, auc, f1_score
import argparse
import regex as re
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', nargs='?', type=str, default='../Tests/<filename>_test', help='Path to test log')

args = parser.parse_args()
path = args.path

## Function that extracts the predicted labels, the true labels, and the certainties of the predictions
def get_labels(file):
	log = open(file, "r")
	text = log.read()

	pattern = "\d+\.\d+ - \d+\.\d+"
	results = re.findall(pattern, text)
	lab_pred = []
	lab_true = []
	for line in results:
		labs = line.split(" - ")
		lab_pred.append(int(float(labs[0])))
		lab_true.append(int(float(labs[1])))

	pattern_prob = "Prob: \d+\.\d+"
	results = re.findall(pattern_prob, text)
	probs = []
	for line in results:
		prob = line.split(" ")
		probs.append(float(prob[1]))
	log.close()
	return lab_pred, lab_true, probs

## Function that maps all certainties from [0,1] to [0.5,1]
def certainties(labels_true, labels_pred, probs):
	right = []
	wrong = []

	for i, p in enumerate(probs):
		lt = labels_true[i]
		lp = labels_pred[i]
		if lt == lp:
			if lt == 0:
				right.append(1 - p)
			else:
				right.append(p)
		else:
			if lt == 0:
				wrong.append(p)
			else:
				wrong.append(1 - p)

	return right, wrong

path_comb = "{}.txt".format(path)
path0 = "{}_t0.txt".format(path)
path1 = "{}_t1.txt".format(path)
path2 = "{}_t2.txt".format(path)

lab_pred, lab_true, probs = get_labels(path_comb)
lab_pred0, lab_true0, probs0 = get_labels(path0)
lab_pred1, lab_true1, probs1 = get_labels(path1)
lab_pred2, lab_true2, probs2 = get_labels(path2)

false_positive_rate0, true_positive_rate0, thresholds0 = roc_curve(lab_true0, probs0)
roc_auc0 = auc(false_positive_rate0, true_positive_rate0)

false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(lab_true1, probs1)
roc_auc1 = auc(false_positive_rate1, true_positive_rate1)

false_positive_rate2, true_positive_rate2, thresholds2 = roc_curve(lab_true2, probs2)
roc_auc2 = auc(false_positive_rate2, true_positive_rate2)

acc = accuracy_score(lab_true, lab_pred)

conf_mat0 = confusion_matrix(lab_true0, lab_pred0)
acc0 = accuracy_score(lab_true0, lab_pred0)
rec0 = recall_score(lab_true0, lab_pred0, average=None)
pre0 = precision_score(lab_true0, lab_pred0, average=None)

conf_mat1 = confusion_matrix(lab_true1, lab_pred1)
acc1 = accuracy_score(lab_true1, lab_pred1)
rec1 = recall_score(lab_true1, lab_pred1, average=None) 
pre1 = precision_score(lab_true1, lab_pred1, average=None) 

conf_mat2 = confusion_matrix(lab_true2, lab_pred2)
acc2 = accuracy_score(lab_true2, lab_pred2)
rec2 = recall_score(lab_true2, lab_pred2, average=None)
pre2 = precision_score(lab_true2, lab_pred2, average=None) 

## Print performance on all tasks

print("\nTask 0")
print("\n\tConfusion_matrix:")
print(conf_mat0)
print("\n\tAccuracy:")
print("%.2f" % (100.0*acc0))
print("\n\tRecall:")
print(rec0)
print("\n\tPrecision:")
print(pre0)
print("\n\tAUC:")
print("%.2f" % (100.0*roc_auc0))

print("\nTask 1")
print("\n\tConfusion_matrix:")
print(conf_mat1)
print("\n\tAccuracy:")
print("%.2f" % (100.0*acc1))
print("\n\tRecall:")
print(rec1)
print("\n\tPrecision:")
print(pre1)
print("\n\tAUC:")
print("%.2f" % (100.0*roc_auc1))


print("\nTask 2")
print("\n\tConfusion_matrix:")
print(conf_mat2)
print("\n\tAccuracy:")
print("%.2f" % (100.0*acc2))
print("\n\tRecall:")
print(rec2)
print("\n\tPrecision:")
print(pre2)
print("\n\tAUC:")
print("%.2f" % (100.0*roc_auc2))

print("\nCombined")
print("\n\tAccuracy:")
print("%.2f" % (100.0*acc))

## PLOT AUROCS

false_positive_rate = [false_positive_rate0,false_positive_rate1,false_positive_rate2]
true_positive_rate = [true_positive_rate0,true_positive_rate1,true_positive_rate2]

roc_t0_col = (255/255.0, 127/255.0, 14/255.0)
roc_t1_col = (44/255.0, 160/255.0, 44/255.0)
roc_t2_col = (214/255.0, 39/255.0, 40/255.0)

roc_cols = [roc_t0_col,roc_t1_col,roc_t2_col]

all_titles = ['Task 1', 'Task 2', 'Task 3']

fig = plt.figure(figsize=(9, 3.7))  
for i in range(3):
	ax = plt.subplot(1,3,i+1)
	ax.set_anchor('S')
	ax.spines["top"].set_visible(False)    
	ax.spines["bottom"].set_visible(False)    
	ax.spines["right"].set_visible(False)    
	ax.spines["left"].set_visible(False)
	ax.plot(false_positive_rate[i], true_positive_rate[i], color = roc_cols[i], label =  'ROC')
	ax.plot(np.arange(100)/100.0,np.arange(100)/100.0, color = 'black', label =  'Random')

	ax.set_xticks(np.arange(0.0, 1.01, 0.1))
	ax.set_yticks(np.arange(0.0, 1.01, 0.1))
	if i == 0:
		plt.ylabel("True positive rate")
		plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)
	else: 
		plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=False) 
	plt.xlabel("False positive rate")
	plt.grid(color="black", alpha=0.2, linestyle='--')
	plt.legend(title = all_titles[i], bbox_to_anchor=(0., 1.02, 1., .102), loc=8, ncol=2, borderaxespad=0.)
fig.tight_layout()
plt.savefig("../AUROC.png", bbox_inches="tight")



