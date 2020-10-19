import math

def get_min_max(imgCount, training_split, split, current_split, lastIndex):
    percent = imgCount * training_split
    minVal = (percent / split) * (current_split)
    maxVal = minVal + (percent / split)

    return math.floor(minVal), math.floor(maxVal) if not lastIndex else math.ceil(maxVal)

def append_to_lists(index, img_per_class, training_split, split, current_split, srcImg, srcLabel, outImg, outLabel, lastIndex=False):
    minVal, maxVal = get_min_max(img_per_class, training_split, split, current_split, lastIndex)

    minVal += index
    maxVal += index

    #print(f"{img_per_class}, min: {minVal}, max: {maxVal}")

    for j in range(minVal, maxVal):
        outImg.append(srcImg[j])
        outLabel.append(srcLabel[j])
    

def lazy_split(img_dataset:list, img_labels:list, images_per_class, split, current_split, lastIndex, training_split:float=.7, shuffle:bool=True)->tuple:
    minIndex = 0
    maxIndex = 0

    train_set = []
    train_label = []

    val_set = []
    val_label = []
    for i in range(len(images_per_class)):
        if i != 0:
            minIndex += images_per_class[i - 1]
        maxIndex += math.floor(images_per_class[i - 1 if i > 0 else 0] * training_split)

        append_to_lists(minIndex, images_per_class[i], training_split, split, current_split, img_dataset, img_labels, train_set, train_label, lastIndex)
        append_to_lists(maxIndex, images_per_class[i], 1 - training_split, split, current_split, img_dataset, img_labels, val_set, val_label)

    #if shuffle:
    #    val_set, val_label = Shuffle(val_set, val_label)
    #    train_set, train_label = Shuffle(train_set, train_label)

    return train_set, train_label, val_set, val_set