import os
import re
import math
import shutil
from PIL import Image
import fnmatch
import random

#Remove all german from ETSDS
#Remove duplicates from GTSD
#Add GTSD to adjusted ETSD in a split way

class AdjustETSD:
    def __init__(self, new_ds_path='Dataset/ETSD_Adjusted/', german_extraction_path='Dataset/ETSD_GTSRB_N_GTSDB/', path_extensions = ['Training/', 'Testing/'], split=0.7, k=3, risky_shuffle=False):
        self.german_extraction_path = german_extraction_path
        self.new_ds_path = new_ds_path
        self.path_extensions = path_extensions
        # Identifiers either default or added by function: extract_german_from_ETSD
        self.GTSRB_training_identifier = '0'
        self.GTSRB_test_identifier = 'TEST_'
        self.GTSDB_training_identifier = 'GTSDB_'
        self.GTSDB_testing_identifier = 'TEST_GTSDB_'

        # parameters
        self.split = split
        self.k = k
        self.risky_shuffle = risky_shuffle


    def delete_sub_ds(self, beginsswith='G_'):
        """Deletes a subdataset by assuming that all of the images starts with a
            specific substring like the GTSDB is assumed to start with G_
        """

        for sub_path in self.path_extensions:
            full_path = self.new_ds_path + sub_path
            print('Deleting files in all subdirectories starting with ' + beginsswith + ' in path: ' + full_path)
            for dir in os.scandir(full_path):
                if dir.is_dir():
                    for file in os.listdir(dir.path):
                        if file.startswith(beginsswith):
                            os.remove(dir.path+'/'+file)
                else:
                    print("is not a dir - error in data")
        print("Finished deleting german data")

    def extract_german_from_ETSD(self, beginsswith = 'G_', path_to_original='ETSD/'):
        """Extract and place german images in other folder by class"""

        os.makedirs(self.german_extraction_path, exist_ok=True)

        for i, sub_path in enumerate(self.path_extensions):
            test_identifier = '' if i == 0 else 'TEST_'
            full_path = path_to_original + sub_path
            print('Extracting and copying in all subdirectories starting with ' + beginsswith + ' in path: ' + path_to_original)
            print('Copying into ' + self.german_extraction_path)
            for dir in os.scandir(full_path):
                if dir.is_dir():
                    class_name = dir.path[-3:]
                    os.makedirs(self.german_extraction_path+'/'+class_name, exist_ok=True)
                    for file in os.listdir(dir.path):
                        if file.startswith(beginsswith):
                            dest_path = self.german_extraction_path + '/' + class_name + '/' + test_identifier + file[2:]
                            image_path = dir.path + '/' + file
                            shutil.copy(image_path, dest_path)
                else:
                    print("is not a dir - error in data")
        print("Finished all the extrating")

    def duplicate_ds(self, path_to_original='ETSD/'):
        """Duplicates a folder to a new path/folder"""
        print("Duplicating folder " + path_to_original + " into -> " + self.new_ds_path)
        print("This can take a while, check your folder to see status")
        shutil.copytree(path_to_original, self.new_ds_path, dirs_exist_ok=False)
        print("Copying finished")

    def count_unique_images(self, dir_list):
        GTSRB_train_d = dict()
        unique_images = 0

        for image in dir_list:
            if image.startswith(self.GTSRB_training_identifier):
                # handle GTSRB duplicate
                unique_sign_key = int(image[:5])
                if unique_sign_key in GTSRB_train_d:
                    GTSRB_train_d[unique_sign_key] = GTSRB_train_d[unique_sign_key] + 1
                else:
                    GTSRB_train_d[unique_sign_key] = 1
            else:
                # asumme other have no duplicate structure
                unique_images += 1
        #Add the unique signs to the unique image counter
        unique_images += len(GTSRB_train_d.keys())

        return unique_images, GTSRB_train_d


    def compute_split(self, unique_image_count, split=0.7):
        return math.ceil(unique_image_count * split)

    def complex_reshuffle(self, class_amount=164, path_to_original='ETSD/'):
        """Reshufle entire ETSD, assume belgian and GTSRB_Train have duplicates, handle those cases.
           Assume all other have NO duplicates as per preliminary analysis.
        """

        padding = '000'
        # german re
        re_GTSRB_train = re.compile('G_\d\d\d\d\d_\d\d\d\d\d.ppm')
        re_Belgian = re.compile('B_\d\d\d\d\d_\d\d\d\d\d.ppm')
        for i in range(1, class_amount):
                print("Writing still - be patient")
                print("class " + str(i))
                #compute class
                class_name = padding[:-len(str(i))] + str(i)
                #handle training
                path_to_training = os.path.join(path_to_original, self.path_extensions[0], class_name)
                GTSRB_training_l, belgian_l, rest_train_l = self.split_into_distinct(path_to_training, self.path_extensions[0], re_GTSRB_train, re_Belgian)
                #handle test
                path_to_testing = os.path.join(path_to_original, self.path_extensions[1], class_name)
                g, b, rest_test_l = self.split_into_distinct(path_to_testing, self.path_extensions[1], re_GTSRB_train, re_Belgian)

                GTSRB_training_l.extend(g)
                belgian_l.extend(b)
                #Sort
                GTSRB_training_l.sort()
                belgian_l.sort()

                # Convert duplicate lists to dicts
                GTSRB_train_d = self.complex_dictify(GTSRB_training_l)
                belgian_d = self.complex_dictify(belgian_l)

                # split and place dictified dups
                self.place_dictified_dups(path_to_original, class_name, GTSRB_train_d)
                self.place_dictified_dups(path_to_original, class_name, belgian_d)

                self.place_simple_imgs(rest_train_l, rest_test_l, class_name)

    def place_simple_imgs(self, train_l, test_l, class_name):
        if self.risky_shuffle:
            train_l.extend(test_l)
            unique_image_count = len(train_l)
            train_count = self.compute_split(unique_image_count, split=self.split)
            random.shuffle(train_l)
            self.place_simple_imgs_subset(train_l[:train_count], class_name, self.path_extensions[0])
            self.place_simple_imgs_subset(train_l[train_count:], class_name, self.path_extensions[1])
        else:
            train_l.extend(test_l)
            unique_image_count = len(train_l)
            train_count = self.compute_split(unique_image_count, split=self.split)
            self.place_simple_imgs_subset(train_l[:train_count], class_name, self.path_extensions[0])
            self.place_simple_imgs_subset(train_l[train_count:], class_name, self.path_extensions[1])

    def place_simple_imgs_subset(self, imgs_l, class_name, path_extension):
        for img in imgs_l:
            src = img
            folder_n_img = src.split('/')[-2:]
            img_only_name = folder_n_img[1]
            dst = self.compute_placement_location(class_name, path_extension, img_only_name, identifier='')
            dst_s = dst.split('/')
            dir_path = os.path.join(dst_s[0], dst_s[1], dst_s[2], dst_s[3])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            shutil.copy(src, dst)

    def place_dictified_dups(self, path_to_original, class_name, dictified_dups_d):
        """Placing GTSRB_train images"""
        if dictified_dups_d:
            index_to_keys = list(dictified_dups_d.keys())
            unique_image_count = len(index_to_keys)
            train_count = self.compute_split(unique_image_count, split=self.split)

            self.place_dictified_dups_subset(index_to_keys[:train_count], dictified_dups_d, path_to_original, self.path_extensions[0], class_name)
            self.place_dictified_dups_subset(index_to_keys[train_count:], dictified_dups_d, path_to_original, self.path_extensions[1], class_name)

    def place_dictified_dups_subset(self, index_to_keys, dictified_dups_d, path_to_original, path_extension, class_name):
        k = self.k
        for key in index_to_keys:
            imgs = dictified_dups_d[key]
            num_of_dups = len(imgs)
            tmp_k = num_of_dups if k > num_of_dups else k  #choose k or  whatever amount below k, which is available
            for img_id in range(num_of_dups-tmp_k, num_of_dups):
                src = imgs[img_id]
                folder_n_img = src.split('/')[-2:]
                img_only_name = folder_n_img[1]
                dst = self.compute_placement_location(class_name, path_extension, img_only_name, identifier='')
                dst_s = dst.split('/')
                dir_path = os.path.join(dst_s[0], dst_s[1], dst_s[2], dst_s[3])
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                shutil.copy(src, dst)

    def complex_dictify(self, imgs_list):
        dictified = dict()

        for img in imgs_list:
            if (key:= int(img.split('_')[1])) in dictified:
                dictified[key].append(img)
            else:
                dictified[int(img.split('_')[1])] = [img]

        return dictified


    def split_into_distinct(self, path_to_class, original_subpath_extension, re_GTSRB_train, re_Belgian):
        """ split into belgian, german and rest"""
        GTSRB_train = list()
        belgian = list()
        rest = list()

        if os.path.exists(path_to_class):
            images = fnmatch.filter(os.listdir(path_to_class), '*.ppm')
            for file_name in images:
                name_with_extension = os.path.join(path_to_class, file_name)
                if re_GTSRB_train.search(file_name):
                    GTSRB_train.append(name_with_extension)
                elif re_Belgian.search(file_name):
                    belgian.append(name_with_extension)
                else:
                    rest.append(name_with_extension)

        return GTSRB_train, belgian, rest


    def re_add_german_to_adjusted_ds(self):
        # Delete orginally placed german images from ETSD_Adjusted
        self.delete_sub_ds()
        # Start re-sprinkling
        print("Re-adding the german now with only k duplicates into folder: " + self.new_ds_path)
        print("This can take a while, check folder for live progress. Will notify once done.")
        for dir in os.scandir(self.german_extraction_path):
            class_name = dir.path[-3:]
            split = self.split
            k = self.k

            dir_list = os.listdir(dir.path)
            if dir_list: # not empty
                GTSRB_imgs, simple_imgs = self.partion_images(dir_list)
                self.handle_GTSRB_train_imgs(class_name, dir.path, GTSRB_imgs, k=k, split=split) #handle GTSRB training set images, which have duplicates
                self.handle_simple_imgs(class_name, dir.path, simple_imgs, split=split)
            print("Finished reading for class --> " + dir.path[-3:])
        print("All german images re-added. Now only with k-duplicates.")
        print("New Dataset ready in -> " + self.new_ds_path)


    def partion_images(self, imgs_in_dir):
        re_GTSRB_train = re.compile('\d\d\d\d\d_\d\d\d\d\d.ppm')
        GTSRB_imgs = list()
        simple_imgs = list()
        for file_name in imgs_in_dir:
            if re_GTSRB_train.search(file_name):
                GTSRB_imgs.append(file_name)
            else:
                simple_imgs.append(file_name)
        return GTSRB_imgs, simple_imgs


    def handle_simple_imgs(self, class_name, path_to_dir, simple_imgs, split=0.7):
        """Place general images, which are assumed unique as per analysis."""
        unique_image_count, _ = self.count_unique_images(simple_imgs)
        training_amount = self.compute_split(unique_image_count)

        if simple_imgs: # not empty
            #Add to training
            self.handle_simple_imgs_sub_range(simple_imgs[:training_amount], class_name, path_to_dir, self.path_extensions[0])
            #Add to testing
            self.handle_simple_imgs_sub_range(simple_imgs[training_amount:], class_name, path_to_dir, self.path_extensions[1])



    def handle_simple_imgs_sub_range(self, simple_imgs, class_name, path_to_dir, path_extension):

        self.folder_exists_or_create(path_extension, class_name)

        for img_file_name in simple_imgs:
            src = path_to_dir+'/'+img_file_name
            dst = self.compute_placement_location(class_name, path_extension, img_file_name)
            shutil.copy(src, dst)


    def handle_GTSRB_train_imgs(self, class_name, path_to_dir, GTSRB_imgs, k=3, split=0.7):
        """Place GTSRB_train images in training and testing set, choose to keep k highest res
            duplicates per unique sign. Split into training and testing by split."""
        unique_image_count, GTSRB_train_d = self.count_unique_images(GTSRB_imgs)
        training_amount = self.compute_split(unique_image_count)
        print(GTSRB_train_d)
        if GTSRB_train_d: # not empty
            #index keys to match number of unique images, as sometimes subsets of classes are split and inserted without correlation to the german dataset
            index_to_keys = list()
            accumulating_end_pivot = [0]
            accumulation = 0
            for key, value in GTSRB_train_d.items():
                index_to_keys.append(key)
                accumulation += value
                accumulating_end_pivot.append(accumulation)

            # sort GTSRB_imgs to ensure that highest k images can be chosen
            GTSRB_imgs.sort()

            # Add to training
            self.GTSRB_train_place_sub_range(0, training_amount, class_name, path_to_dir, index_to_keys, accumulating_end_pivot, GTSRB_imgs, GTSRB_train_d, k, self.path_extensions[0])
            # Add to testing
            self.GTSRB_train_place_sub_range(training_amount, unique_image_count, class_name, path_to_dir, index_to_keys, accumulating_end_pivot, GTSRB_imgs, GTSRB_train_d, k, self.path_extensions[1])

    def GTSRB_train_place_sub_range(self, start, end, class_name, path_to_dir, index_to_keys, accumulating_end_pivot, GTSRB_imgs, GTSRB_train_d, k, path_extension):

        self.folder_exists_or_create(path_extension, class_name)

        for x in range(start, end):

            img_id = index_to_keys[x]
            num_of_dups = GTSRB_train_d[img_id]

            tmp_k = num_of_dups if k > num_of_dups else k  #choose k or  whatever amount below k, which is available

            # compute the specific k-highest list then use that
            k_r_start = accumulating_end_pivot[x+1] - tmp_k
            k_r_end = accumulating_end_pivot[x+1]
            k_selected_imgs = GTSRB_imgs[k_r_start:k_r_end]

            # Add images
            for img_file_name in k_selected_imgs:
                src = path_to_dir+'/'+img_file_name
                dst = self.compute_placement_location(class_name, path_extension, img_file_name)
                shutil.copy(src, dst)

    def folder_exists_or_create(self, path_extension, class_name):
        """Check if folder exists if not, then create folder, as the resampling of german ds, might
            create a new folder in the dataset which previously had no examples"""
        folder_placement = self.new_ds_path + path_extension + class_name
        if not os.path.exists(folder_placement):
            os.makedirs(folder_placement, exist_ok=False)

    def compute_placement_location(self, class_name, extension, img_file_name , identifier='G_'):
        """Compute the path for where to place a specific image depending on its class and
            whether it is training or testing"""
        return self.new_ds_path + extension + class_name + '/' + identifier+img_file_name

    def trim_imgs_according_to_predicate(self, predicate, path=None, subpath_extensions=None, file_extension='.ppm'):
        """Runs through the dataset training and test, opens the image as pil objects,
            and removes all files, which obeys the predicate"""
        # Alow to specify other paths than default
        path = path or self.new_ds_path
        subpath_extensions = subpath_extensions or self.path_extensions
        num_of_deleted_images = 0
        for sub_path in subpath_extensions:
            full_path = path+sub_path
            print('Trimming dataset in --> ' + full_path + ' according to predicate received.')
            for dir in os.scandir(full_path):
                if dir.is_dir():
                    for file in os.listdir(dir.path):
                        if file.endswith(file_extension):
                            path_to_file = dir.path+'/'+file
                            Im = Image.open(path_to_file)
                            if predicate(Im): #Delete if image obeys predicate
                                Im.close() # Close, so it can be destroyed
                                os.remove(path_to_file)
                                num_of_deleted_images +=1
                else:
                    print("is not a dir - error in data")
        print("Dataset trimmed by: " + str(num_of_deleted_images) + " as reqeusted through received predicate.")
        print("Dataset ready in -> " + self.new_ds_path)

def run_milad():
    full_path_to_original_ETSD = 'Dataset/milad_gains_images/' # must be changed to match full path to original
    a = AdjustETSD(split=0.7, k=3, path_extensions = ['Training/', 'Testing/']) # split for training / test | k = highest res duplicates
    # Create copy for security and redundancy
    # will be here --> ETSD_Adjusted
    if not os.path.exists(a.new_ds_path):
        a.duplicate_ds(full_path_to_original_ETSD)
    # Extract german images to here --> ETSD_GTSRB_N_GTSDB
    if not os.path.exists(a.german_extraction_path):
        a.extract_german_from_ETSD(path_to_original=full_path_to_original_ETSD)
    # Sprinkle german images into ETSD_Adjusted by split, and by k highes resolution images
    a.re_add_german_to_adjusted_ds()
    # Remove images according to lambda predicates, easily extensible with more predicates on the .ppm pil image
    # Default it operates on the ETSD_Adjusted, with subpath_extensions =['Training/', 'Testing/']
    # Can be overriden (check function above)
    width_threshold = 10
    height_threshold = 10
    a.trim_imgs_according_to_predicate(lambda pil_image: pil_image.width < width_threshold or pil_image.height < height_threshold)

def complex_runner(path_to_original='ETSD/'):
    a = AdjustETSD(split=0.7, k=3, path_extensions=['Training/', 'Testing/'], new_ds_path='Dataset/ETSD_Adjusted/', risky_shuffle=False)
    print("Deleting if old Adjustment exists. - Be patient...")
    if os.path.exists(a.new_ds_path):
        shutil.rmtree(a.new_ds_path)
    print("Analyzing structure.")
    print("Recursive analysis finished.")
    print("Preparing structured filter write of new ds.")
    shutil.copytree(path_to_original, a.new_ds_path, copy_function=lambda src, dst: shutil.copy2(src,dst) if os.path.isdir(src) else None)
    a.complex_reshuffle(class_amount=164, path_to_original=path_to_original)
    print("Writing completed.")
    print("Preparing trim thresholds for dynamic post filtering. - Stay patient")
    width_threshold = 10
    height_threshold = 10
    a.trim_imgs_according_to_predicate(lambda pil_image: pil_image.width < width_threshold or pil_image.height < height_threshold)

if __name__ == "__main__":
    # run_milad()
    # a = AdjustETSD()
    # a.complex_reshuffle()
    complex_runner()
