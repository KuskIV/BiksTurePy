import os
import csv
import fnmatch
import random
import shutil
import pandas as pd

from sklearn.model_selection import train_test_split
import numpy

def rename_folder(bjarke_dumb, c, rn):
  try:
      os.rename(bjarke_dumb + "/" + 'Testing' + "/" + str(c), bjarke_dumb + "/" + "Testing" + "/" + str(rn))
      os.rename(bjarke_dumb + "/" + "Training" + "/" + str(c), bjarke_dumb + "/" + "Training" + "/" + str(rn))
  except:
      print('Error in rename.')

# Move files
def move_files(files, bjarke_dumb, c, tFile):
  for j in files:
    try:
      shutil.move(j, bjarke_dumb + "/" + tFile + "/" + c)
    except:
      pass

# Delete if below a certain amount of samples
def remove_folder(bjarke_dumb, c):
  test_path = bjarke_dumb + "/" + 'Testing' + "/" + c
  if os.path.exists(test_path):
    shutil.rmtree(bjarke_dumb + "/" + 'Testing' + "/" + c)
  
  train_path = bjarke_dumb + "/" + 'Training' + "/" + c
  if os.path.exists(train_path):
    shutil.rmtree(bjarke_dumb + "/" + 'Training' + "/" + c)

  # try:
  #   shutil.rmtree(path + "/" + 'Testing' + "/" + c)
  #   shutil.rmtree(path + "/" + 'Training' + "/" + c)
  # except:
  #   print("ERROR")

def add_folder(bjarke_dumb, tdir, c):
  save_path = bjarke_dumb + "/" + tdir + "/" + c
  if not os.path.exists(save_path):
    os.makedirs(bjarke_dumb + "/" + tdir + "/" + c)

  # try:
  #   os.makedirs(path + "/" + tdir + "/" + c)
  # except:
  #   print("ERROR")

# Method to make temporary folder
def make_temporary_folder(bjarke_dumb, name):
  if not os.path.isdir(bjarke_dumb + "/temp/" + name):
    os.makedirs(bjarke_dumb + "/temp/" + name)

# Split the list of classes
def randomize_and_split_list(slist, bjarke_dumb, c, testing_percentage):
  #random.shuffle(slist)
  slist = numpy.array(slist)
  x_train, x_test = train_test_split(slist, test_size=testing_percentage)

  move_files(x_train, bjarke_dumb, c, 'Training')
  move_files(x_test, bjarke_dumb, c, 'Testing')

# Compare two lists and return list with matches
def get_classes_from_folders(bjarke_dumb):
  # Get folders to check for classes
  rootList = os.listdir(bjarke_dumb)
  classes = {}
  for i in rootList:
   if(os.path.isdir(bjarke_dumb + "/" + i)):
    cList = os.listdir(bjarke_dumb + "/" + i)
    cList = [int(i) for i in cList]
    cList.sort()
    cList = [str(i).zfill(3) for i in cList]
    # print(cList)
    for h in cList:
      path_from_list = bjarke_dumb + "/" + i + "/" + str(h)
      if(os.path.isdir(path_from_list)):
        if h in classes:
          classes[h].append(path_from_list)
        else:
          classes[h] = [path_from_list]
  return classes

def get_samples_from_folders(classes, testing_percentage, bjarke_dumb):
  for c in classes:
    split = []
    for j in classes[c]:
      tList = [j + "/" + s for s in fnmatch.filter(os.listdir(j), '*.ppm')]
      for i in tList:
       split.append(i)

    if len(split) < 10:
      remove_folder(bjarke_dumb, c)
    else:
      add_folder(bjarke_dumb, 'Training', c)
      add_folder(bjarke_dumb, 'Testing', c)
      randomize_and_split_list(split, bjarke_dumb, c, testing_percentage)

def trim_classes(classes):
  sList = list()
  for key in classes:
    if str(key).lstrip('0') != '':
        sList.append(str(key).lstrip('0'))
    else:
        sList.append('0')
  sList =  [int(i) for i in sList]
  return sList

def find_and_edit(old, bjarke_dumb):
  classes = trim_classes(get_classes_from_folders(bjarke_dumb))
  classes = [str(i) for i in classes]
  lines = list()
  count = 0
  tmp_cell = ""

  with open(old + '/Classes_Description.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    for row in reader:
      if row[4] in classes:
        if tmp_cell != "":
          row[2] = tmp_cell
          tmp_cell = ''
        row[4] = count
        count += 1
        lines.append(row)
      else:
        if row[4] != 'European class':
          if row[2] != '':
            tmp_cell = row[2]
          # row.remove()
  return lines

def save_csv_file(bjarke_dumb, lines):
  with open(bjarke_dumb + '/test.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)

def rename_folders(bjarke_dumb):
  clist = get_classes_from_folders(bjarke_dumb)
  # print(clist)
  count = 0
  for c in clist:
        rename_folder(bjarke_dumb,c,count)
        count += 1

def run_split_dataset(bjarke_dumb, testing_percentage, old):
  clist = get_classes_from_folders(bjarke_dumb)
  get_samples_from_folders(clist, testing_percentage, bjarke_dumb)
  save_csv_file(bjarke_dumb, find_and_edit(old, bjarke_dumb))
  rename_folders(bjarke_dumb)

#path = r"C:/Users\bbalt/Desktop/European Traffic Sign Dataset"


#rootList = os.listdir(path)
#classes = {}
#classList = []

# Get folders to check for classes
#for i in rootList:
#  if(os.path.isdir(path + "/" + i)):
#    for h in os.listdir(path + "/" + i):
#      if(os.path.isdir(path + "/" + i + "/" + h)):
#        if h in classes:
#          classes[h].append(path + "/" + i + "/" + h)
#        else:
#          classes[h] = [path + "/" + i + "/" + h]

# Move samples and split
#for c in classes:
#  split = []
#  for j in classes[c]:
#    tList = [j + "/" + s for s in fnmatch.filter(os.listdir(j), '*.ppm')]
#    for i in tList:
#      split.append(i)

#  if len(split) < 10:
#    remove_folder(path, c)
#  else:
#    add_folder(path, 'Training', c)
#    add_folder(path, 'Testing', c)
#    randomize_and_split_list(split, path, c)


# Get CSV file from each class and combine csv.

#for i in classes:
#  x = 0
#  for h in classes[i]:
#    for j in os.listdir(h):
#      if j.endswith('.csv'):
#        x += 1

# Compare folders and take classes that matches
#prevDir = ''
#for key in dirList:
#  if prevDir != '':
#    classList = find_matches_in_list(dirList[prevDir], dirList[key])
#  else:
#    prevDir = key

# Create temporary class folders
#for i in classList:
#  make_temporary_folder(path, i)

#test = [i for i, j in zip(dirList[0], dirList[1]) if i == j]

