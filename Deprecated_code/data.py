# def get_ppm_pictures(path):
#     dataset_placements = []
#     images_per_class = []

#     for i in range(43):
#         name = str(i).zfill(5)
#         num_of_images = 0
#         for files in os.listdir(path + "/" + name):
#             if files.endswith(".ppm"):
#                 dataset_placements.append([path + "/" + name + "/" +  files, int(os.path.basename(name))])
#                 num_of_images += 1
#         images_per_class.append(num_of_images)

#     return dataset_placements, images_per_class

# def get_data(fixed_size:tuple=(0,0), padded_images:bool = False, smart_resize:bool = True)->tuple:
#     # extract data from raw
    

#     raw_dataset, images_per_class = get_ppm_pictures(get_dataset_path())

#     if padded_images:
#         print("Padded images not implemented yet, only resize and smart resize.")

#     # convert ppm to numpy arrays
#     numpy_images = convert_imgs_to_numpy_arrays(raw_dataset)
#     # auto reshape images
#     numpy_images_reshaped = auto_reshape_images(fixed_size, numpy_images, smart_resize)

#     # get labels for each training example in correct order
#     labels = get_labels(raw_dataset)

#     return numpy_images_reshaped, labels, images_per_class

# def update_values(i, images_per_class, label_index, training_split):
#     maxVal = images_per_class[label_index] + i
#     pictures_in_current_class = images_per_class[label_index]
#     dist_in_current_class = math.ceil(pictures_in_current_class * training_split)

#     return maxVal, dist_in_current_class, 0


# def split_data(img_dataset:list, img_labels:list, images_per_class, training_split:float=.7, shuffle:bool=True)->tuple: # Migth not be used anymore, once lazt is implemented
#     """Input numpy array of images, numpy array of labels.
#        Return a tuple with (training_images, training_labels, test_images, test_labels).
#        Does have stochastic/shuffling of the data with shuffle parameter."""


#     train_set = []
#     train_label = []

#     val_set = []
#     val_label = []

#     label_index = 0
#     maxVal, dist_in_current_class, val_in_train = update_values(0, images_per_class, label_index, training_split)
#     for i in range(len(img_dataset)):
#         if i > maxVal:
#             label_index += 1
#             if not (label_index > len(images_per_class)):
#                 maxVal, dist_in_current_class, val_in_train = update_values(i, images_per_class, label_index, training_split)
#         if val_in_train < dist_in_current_class:
#             train_label.append(img_labels[i])
#             train_set.append(img_dataset[i])
#             val_in_train += 1
#         else:
#             val_label.append(img_labels[i])
#             val_set.append(img_dataset[i])

#     if shuffle:
#         val_set, val_label = Shuffle(val_set, val_label)
#         train_set, train_label = Shuffle(train_set, train_label)

#     return train_set, train_label, val_set, val_label