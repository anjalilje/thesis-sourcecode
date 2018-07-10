## Load packages
import regex as re
import argparse
from glob import glob

## Function that replaces nans in a list by the previous value in the list
def check_nan(result, prev):
    if float(result[0]) < 0.0000001:
        result = prev
    return result

## Function that extracts the accuracies from the output logs of the training sessions
def get_accuracy(text, kind = "training"):
    text = re.sub("nan", "0.0", text)
    pattern = "Average {} accuracy\: \d+\.\d+".format(kind)
    results = re.findall(pattern, text)
    accs = []
    prev = 0.0
    for line in results:
        acc = re.findall("\d+\.\d+", line)
        acc = check_nan(acc, prev)
        prev = acc
        accs.append(acc[0])
    return accs

## Function that extracts the losses from the output logs of the training sessions
def get_loss(text, kind = "training"):
    text = re.sub("nan", "0.0", text)
    pattern = "Average {} loss\: \d+\.\d+".format(kind)
    results = re.findall(pattern, text)
    loss = []
    prev = 0.0
    for line in results:
        l = re.findall("\d+\.\d+", line)
        l = check_nan(l, prev)
        prev = l
        loss.append(l[0])
    return loss

## Function that combines the logs of 1 or more training sessions – in case training has been stopped and continued in another session
def combine_logs(files):
    train_accs = []
    valid_accs = []
    train_loss = []
    valid_loss = []
    for file in files:
        log = open(file, "r")
        text = log.read()
        train_acc = get_accuracy(text, kind = "training")
        valid_acc = get_accuracy(text, kind = "validation")
        train_l = get_loss(text, kind = "training")
        valid_l = get_loss(text, kind = "validation")

        train_accs += train_acc
        valid_accs += valid_acc
        train_loss += train_l
        valid_loss += valid_l

        log.close()

    return train_accs, valid_accs, train_loss, valid_loss


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', nargs='?', type=str, default='../Logs', help='Path to logs')
parser.add_argument('-nn', '--net_name', nargs='?', type=str, default='vgg16', help='Name of network')

args = parser.parse_args()
path = args.path
net_name = args.net_name

files = glob("{}/*{}.txt".format(path, net_name))

files.sort()

train_accs, valid_accs, train_loss, valid_loss = combine_logs(files)

# There are 4 output files: one for each list of training and validation accuracy and loss
f_train_acc = open("{}/{}-train_accs.txt".format(path, net_name), 'w')
f_valid_acc = open("{}/{}-valid_accs.txt".format(path, net_name), 'w')
f_train_loss = open("{}/{}-train_loss.txt".format(path, net_name), 'w')
f_valid_loss = open("{}/{}-valid_loss.txt".format(path, net_name), 'w')

for train_acc in train_accs:
	print(train_acc, file = f_train_acc)
f_train_acc.close()

for valid_acc in valid_accs:
	print(valid_acc, file = f_valid_acc)
f_valid_acc.close()

for train_l in train_loss:
    print(train_l, file = f_train_loss)
f_train_loss.close()

for valid_l in valid_loss:
    print(valid_l, file = f_valid_loss)
f_valid_loss.close()