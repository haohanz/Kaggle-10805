from collections import defaultdict
import random
random.seed(666)

YOUR_PATH = '../../Open_image_256'

all_labels = open('classes-trainable.csv').readlines()[1:]
label_list = open('train_human_labels.csv').readlines()[1:]
labels_inlist = set([i.split(',')[2] for i in label_list])
all_labels = sorted([i.strip() for i in all_labels if i.strip() in labels_inlist])
print('num_classes:{}'.format(len(all_labels)))


label_dict = {}
k = 0
for label in all_labels:
	label_dict[label] = k
	k += 1


data = defaultdict(list)

for one_data in label_list:
	data_id, _, data_label, _ = one_data.split(',')
	if data_label in label_dict:
		data[data_id].append(label_dict[data_label])


f = open('train.txt','w')
g = open('val.txt','w')
for one_data in data:
	string = YOUR_PATH+'/train_{}_256/'.format(one_data[0])+one_data+'.jpg\t'
	for label in data[one_data]:
		string+=str(label)+','
	string = string[:-1]+'\n'
	if random.random()>0.95:
		g.write(string)
	else:
		f.write(string)

f.close()
g.close()
