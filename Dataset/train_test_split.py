import os
import csv
import fnmatch
import random
import shutil
import pandas as pd

from sklearn.model_selection import train_test_split
import numpy

def rename_folder(path, c, rn):
  try:
    os.rename(path + "/" + 'Testing' + "/" + c, path + "/" + "Testing" + "/" + str(rn))
    os.rename(path + "/" + "Training" + "/" + c, path + "/" + "Training" + "/" + str(rn))
  except:
    pass

# Move files
def move_files(files, path, c, tFile):
  for j in files:
    try:
      shutil.move(j, path + "/" + tFile + "/" + c)
    except:
      pass

# Delete if below a certain amount of samples
def remove_folder(path, c):
  try:
    shutil.rmtree(path + "/" + 'Testing' + "/" + c)
    shutil.rmtree(path + "/" + 'Training' + "/" + c)
  except:
    pass

def add_folder(path, tdir, c):
  try:
    os.makedirs(path + "/" + tdir + "/" + c)
  except:
    pass

# Method to make temporary folder
def make_temporary_folder(path, name):
  if not os.path.isdir(path + "/temp/" + name):
    os.makedirs(path + "/temp/" + name)

# Split the list of classes
def randomize_and_split_list(slist, path, c, testing_percentage):
  random.shuffle(slist)
  slist = numpy.array(slist)
  x_train, x_test = train_test_split(slist, test_size=testing_percentage)
  
  move_files(x_train, path, c, 'Training')
  move_files(x_test, path, c, 'Testing')

# Compare two lists and return list with matches
def get_classes_from_folders(path):
  # Get folders to check for classes
  rootList = os.listdir(path)
  classes = {}
  for i in rootList:
   if(os.path.isdir(path + "/" + i)):
      for h in os.listdir(path + "/" + i):
        if(os.path.isdir(path + "/" + i + "/" + h)):
          if h in classes:
            classes[h].append(path + "/" + i + "/" + h)
          else:
            classes[h] = [path + "/" + i + "/" + h]
  return classes

def get_samples_from_folders(classes, testing_percentage):
  for c in classes:
    split = []
    for j in classes[c]:
      tList = [j + "/" + s for s in fnmatch.filter(os.listdir(j), '*.ppm')]
      for i in tList:
       split.append(i)
    
    if len(split) < 10:
      remove_folder(path, c)
    else:
      add_folder(path, 'Training', c)
      add_folder(path, 'Testing', c)
      randomize_and_split_list(split, path, c, testing_percentage)

def trim_classes(classes):
  sList = list()
  for key in classes:
    sList.append(key.lstrip('0'))
  return sList

def find_and_edit(path):
  classes = trim_classes(get_classes_from_folders(path))
  lines = list()
  count = 0

  with open(path + '/Classes_Description.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    for row in reader:
      if row[5] in classes:
        row[5] = count        
        count += 1
      else:
        if row[5] != 'European class':
          row[5] = 'x'
      lines.append(row)
  return lines

def save_csv_file(path, lines):
  with open(path + '/test.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)

def rename_folders(path):
  clist = get_classes_from_folders(path)
  count = 0
  for i in clist:
    rename_folder(path,i,count)
    count += 1

def run_split_dataset(path, testing_percentage):
  clist = get_classes_from_folders(path)
  get_samples_from_folders(clist, testing_percentage) 
  save_csv_file(path, find_and_edit(path))
  rename_folders(path)

path = r'C:\Users\bbalt\Desktop\satina_gains_images'

run_split_dataset(path, 0.3)


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

