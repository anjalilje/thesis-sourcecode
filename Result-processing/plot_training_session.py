## Load packages
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from glob import glob

## Function that takes the files created by combine_logs.py and puts them in arrays
def file_to_arr(file):
	f = open(file, 'r')
	text = f.read()
	str_accs = text.split("\n")
	accs = []

	for i in range(len(str_accs)-1):
		accs.append(float(str_accs[i]))

	f.close()

	return np.array(accs)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', nargs='?', type=str, default='../Logs', help='Path to logs')
parser.add_argument('-nn', '--net_name', nargs='?', type=str, default='vgg16', help='Name of network')

args = parser.parse_args()
path = args.path
net_name = args.net_name

## Set the early stop variable to the point of the minimum loss during training
early_stop = 71
step = 20

train_acc_file = "{}/{}-train_accs.txt".format(path, net_name)
train_acc_t0_file = "{}/{}_t0-train_accs.txt".format(path, net_name)
train_acc_t1_file = "{}/{}_t1-train_accs.txt".format(path, net_name)
train_acc_t2_file = "{}/{}_t2-train_accs.txt".format(path, net_name)

valid_acc_file = "{}/{}-valid_accs.txt".format(path, net_name)
valid_acc_t0_file = "{}/{}_t0-valid_accs.txt".format(path, net_name)
valid_acc_t1_file = "{}/{}_t1-valid_accs.txt".format(path, net_name)
valid_acc_t2_file = "{}/{}_t2-valid_accs.txt".format(path, net_name)

train_loss_file = "{}/{}-train_loss.txt".format(path, net_name)
train_loss_t0_file = "{}/{}_t0-train_loss.txt".format(path, net_name)
train_loss_t1_file = "{}/{}_t1-train_loss.txt".format(path, net_name)
train_loss_t2_file = "{}/{}_t2-train_loss.txt".format(path, net_name)

valid_loss_file = "{}/{}-valid_loss.txt".format(path, net_name)
valid_loss_t0_file = "{}/{}_t0-valid_loss.txt".format(path, net_name)
valid_loss_t1_file = "{}/{}_t1-valid_loss.txt".format(path, net_name)
valid_loss_t2_file = "{}/{}_t2-valid_loss.txt".format(path, net_name)

train_accs = file_to_arr(train_acc_file)
train_t0_accs = file_to_arr(train_acc_t0_file)
train_t1_accs = file_to_arr(train_acc_t1_file)
train_t2_accs = file_to_arr(train_acc_t2_file)

valid_accs = file_to_arr(valid_acc_file)
valid_t0_accs = file_to_arr(valid_acc_t0_file)
valid_t1_accs = file_to_arr(valid_acc_t1_file)
valid_t2_accs = file_to_arr(valid_acc_t2_file)

train_loss = file_to_arr(train_loss_file)
train_t0_loss = file_to_arr(train_loss_t0_file)
train_t1_loss = file_to_arr(train_loss_t1_file)
train_t2_loss = file_to_arr(train_loss_t2_file)

valid_loss = file_to_arr(valid_loss_file)
valid_t0_loss = file_to_arr(valid_loss_t0_file)
valid_t1_loss = file_to_arr(valid_loss_t1_file)
valid_t2_loss = file_to_arr(valid_loss_t2_file)

n_train_accs = len(train_accs)
n_valid_accs = len(valid_accs)
n_train_loss = len(train_loss)
n_valid_loss = len(valid_loss)

train_col = (31/255.0, 119/255.0, 180/255.0)
valid_col = (174/255.0, 199/255.0, 232/255.0)

train_t0_col = (255/255.0, 127/255.0, 14/255.0)
valid_t0_col = (255/255.0, 187/255.0, 120/255.0)

train_t1_col = (44/255.0, 160/255.0, 44/255.0)
valid_t1_col = (152/255.0, 223/255.0, 138/255.0)

train_t2_col = (214/255.0, 39/255.0, 40/255.0)
valid_t2_col = (255/255.0, 152/255.0, 150/255.0)

all_train_accs = [train_accs, train_t0_accs, train_t1_accs, train_t2_accs]
all_valid_accs = [valid_accs, valid_t0_accs, valid_t1_accs, valid_t2_accs]
all_train_loss = [train_loss, train_t0_loss, train_t1_loss, train_t2_loss]
all_valid_loss = [valid_loss, valid_t0_loss, valid_t1_loss, valid_t2_loss]
all_train_cols = [train_col, train_t0_col, train_t1_col, train_t2_col]
all_valid_cols = [valid_col, valid_t0_col, valid_t1_col, valid_t2_col]
all_titles = ['Combined', 'Task 1', 'Task 2', 'Task 3']

N = early_stop + 25

min_acc = 0.5
max_loss = 0.81

## Plot the combined training and validation accuracy and loss

fig1 = plt.figure(figsize=(6.3, 3))  
ax = plt.subplot(1,2,1)
ax.set_anchor('S')
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)
plt.axvline(x = early_stop, color = 'black', linestyle='--', alpha=0.5)
ax.plot(np.arange(0, N), all_train_accs[0][0:N], color = all_train_cols[0])#, label =  'Training')
ax.plot(np.arange(0, N), all_valid_accs[0][0:N], color = all_valid_cols[0])#, label = 'Validation')

ytix = np.arange(min_acc, 1.01, 0.05)
ax.set_xticks(np.arange(0, N+1, step))
ax.set_yticks(ytix)
ax.set_xlim(-1,N+1)
ax.set_ylim(min_acc,1)
plt.ylabel("Accuracy")
plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)
plt.xlabel("Epoch")
plt.grid(color="black", alpha=0.2, linestyle='--')

ax = plt.subplot(1,2,2)
ax.set_anchor('S')
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)
plt.axvline(x = early_stop, color = 'black', linestyle='--', alpha=0.5)
line1 = ax.plot(np.arange(0, N), np.array(all_train_loss[0][0:N]), color = all_train_cols[0], label =  'Training')
line2 = ax.plot(np.arange(0, N), all_valid_loss[0][0:N], color = all_valid_cols[0], label = 'Validation')

ytix = np.arange(0.0, max_loss, 0.1)
ax.set_xticks(np.arange(0, N+1, step))
ax.set_yticks(ytix)
ax.set_xlim(-1,N+1)
ax.set_ylim(0,max_loss+0.05)
plt.ylabel("Loss")
plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)
plt.xlabel("Epoch")
plt.grid(color="black", alpha=0.2, linestyle='--')
handles, labels = ax.get_legend_handles_labels()
plt.figlegend(handles=handles, labels=labels, title = all_titles[0], bbox_to_anchor=(0., 1.02, 1., .102), loc=8, ncol=2, borderaxespad=0.)

fig1.tight_layout(pad=0.1, w_pad=0.2)
plt.savefig("../combined_accloss.png", bbox_inches="tight")

## Plot the task training and validation accuracies and losses

fig2 = plt.figure(figsize=(9, 6))  
for i in range(1,4):
	ax = plt.subplot(2,3,i)
	ax.set_anchor('S')
	ax.spines["top"].set_visible(False)    
	ax.spines["bottom"].set_visible(False)    
	ax.spines["right"].set_visible(False)    
	ax.spines["left"].set_visible(False)
	plt.axvline(x = early_stop, color = 'black', linestyle='--', alpha=0.5)
	ax.plot(np.arange(0, N), all_train_accs[i][0:N], color = all_train_cols[i], label =  'Training')
	ax.plot(np.arange(0, N), all_valid_accs[i][0:N], color = all_valid_cols[i], label = 'Validation')

	ytix = np.arange(min_acc, 1.01, 0.05)
	ax.set_xticks(np.arange(0, N+1, step))
	ax.set_yticks(ytix)
	ax.set_xlim(-1,N+1)
	ax.set_ylim(min_acc,1)
	if i == 1:
		plt.ylabel("Accuracy")
		plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=True)
	else: 
		plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False) 
	plt.grid(color="black", alpha=0.2, linestyle='--')
	plt.legend(title = all_titles[i], bbox_to_anchor=(0., 1.02, 1., .102), loc=8, ncol=2, borderaxespad=0.)

for i in range(1,4):
	ax = plt.subplot(2,3,i+3)
	ax.set_anchor('S')
	ax.spines["top"].set_visible(False)    
	ax.spines["bottom"].set_visible(False)    
	ax.spines["right"].set_visible(False)    
	ax.spines["left"].set_visible(False)
	plt.axvline(x = early_stop, color = 'black', linestyle='--', alpha=0.5)
	ax.plot(np.arange(0, N), all_train_loss[i][0:N], color = all_train_cols[i], label =  'Training')
	ax.plot(np.arange(0, N), all_valid_loss[i][0:N], color = all_valid_cols[i], label = 'Validation')
	
	ytix = np.arange(0.0, max_loss, 0.1)
	ax.set_xticks(np.arange(0, N+1, step))
	ax.set_yticks(ytix)
	ax.set_xlim(-1,N+1)
	ax.set_ylim(0,max_loss+0.05)
	if i == 1:
		plt.ylabel("Loss")
		plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)
	else: 
		plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=False) 
	plt.xlabel("Epoch")
	plt.grid(color="black", alpha=0.2, linestyle='--')
fig2.tight_layout(pad=0.1, w_pad=0.2, h_pad=1.0)
plt.savefig("../task_accloss.png", bbox_inches="tight")
