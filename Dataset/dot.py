import os
import re
import csv

total_classes = 164
test_sub_path = 'Testing'
training_sub_path = 'Training'
initial_path = 'ETSD/'
class_descrip_csv_path = 'class_descrip.csv'

def count(path):
    re_class_name = re.compile('(?<=\\\\)\d*')
    class_counter_d = dict()

    for dir in os.scandir(path):
        if dir.is_dir():
            num_of_images = 0
            if match := re_class_name.search(dir.path):
                class_name = match.group(0)
            else:
                print('corruption in match with file name')

            for file in os.listdir(dir.path):
                if file.endswith('.ppm'):
                    num_of_images += 1
        else:
            print("is not a dir - error in data")
        class_counter_d[int(class_name)] = [dir.path, num_of_images]


    return class_counter_d


def read_class_descrip(path):
    category = list()
    subcategory = list()
    class_label = list()

    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_label.append(row['Descriptive name'])
            if categoryName:= row['Category']:
                cat_pivot_id = row['European class']
                category.append([categoryName, cat_pivot_id, 0])

                if len(category) > 1: #tricky part
                    if categoryName != category[-2][0]:
                        category[-2][2] = cat_pivot_id

            if subcategoryName:= row['Subcategory']:
                sub_pivot_id = row['European class']
                subcategory.append([subcategoryName, sub_pivot_id, 0])

                if len(subcategory) > 1: #tricky part
                    if subcategoryName != subcategory[-2][0]:
                        subcategory[-2][2] = sub_pivot_id
    #clean up last with end index
    end = len(class_label)+1
    category[-1][2] = end
    subcategory[-1][2] = 160

    return class_label, category, subcategory

def count_categories(counter_d, categories):
    count_categories = dict()
    for category in categories:
        total_per_category = 0
        for i in range(int(category[1]), int(category[2])):
            if i in counter_d:
                total_per_category += counter_d[i][1]

        count_categories[category[0]] = total_per_category

    return count_categories

def combine_dictionaries(training_counter_d, test_counter_d):
    combined_counter_d = training_counter_d
    for key in combined_counter_d:
        if key in test_counter_d:
            combined_counter_d[key][1] += test_counter_d[key][1]
    return combined_counter_d

def write_counting_to_csv(path, count_per_category, fieldnames):

    with open(path, 'w', newline='\n') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key in count_per_category.keys():
            writer.writerow({fieldnames[0]: key, fieldnames[1]: count_per_category[key]})

def get_category_from_class(category_split_list, class_id):
    for category in category_split_list:
        if int(class_id) >= int(category[1]) and int(class_id) < int(category[2]):
            return category[0]
    return None


# if __name__ == '__main__':
#     training_counter_d = count(initial_path+training_sub_path)
#     test_counter_d = count(initial_path+test_sub_path)

#     combined_counter_d = combine_dictionaries(training_counter_d, test_counter_d)

#     read_class_descrip(class_descrip_csv_path)
#     class_text_labels, categories, subcategories = read_class_descrip(class_descrip_csv_path)
#     #Adjusting subcategories
#     subcategories.append(categories[0]) #add danger as sub category
#     subcategories.append(categories[-1]) #add others as sub category

#     count_per_category = count_categories(combined_counter_d, categories)

#     count_per_subcategory = count_categories(combined_counter_d, subcategories)
#     # Add Danger and other to subcategory
#     # count_per_subcategory['Dangerous warning'] = count_per_category['Dangerous warning']
#     # count_per_subcategory['Others'] = count_per_category['Others']
#     #
#     # print('Number of images per category printed to categories.csv')
#     # write_counting_to_csv('categories.csv', count_per_category, ['Category', 'Number of images'])
#     # print('Number of images per category printed to subcategories.csv')
#     # write_counting_to_csv('subcategories.csv', count_per_subcategory, ['Subcategory', 'Number of images'])

#     list_of_categories = count_per_category.keys()
#     list_of_subcategories = count_per_subcategory.keys()

#     print(subcategories)

#     print(get_category_from_class(categories, 164))
#     print(get_category_from_class(subcategories, 1))
