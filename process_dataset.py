# processing the raw data of the video datasets (Something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Bolei Zhou, Dec.2 2017
#
#
import os
import pdb
dataset_name = 'something-something-v1' # 'jester-v1'
with open('%s-labels.csv'% dataset_name) as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories) #排序了
with open('category.txt','w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i
print(dict_categories)
files_input = ['%s-validation.csv'%dataset_name,'%s-train.csv'%dataset_name]
files_output = ['val_videofolder.txt','train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split(';')
        folders.append(items[0]) #视频名字是用数字命名的
        idx_categories.append(dict_categories[items[1]])
    print("idx_categories:",idx_categories)
    print("folders:",folders)
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join('/home/aistudio/data/data30061/20bn-%s'%dataset_name, curFolder))
        output.append('%s %d %d'%(curFolder, len(dir_files), curIDX))
        print('%d/%d'%(i, len(folders)))
    with open(filename_output,'w') as f:
        f.write('\n'.join(output))
