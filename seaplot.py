import re
import matplotlib.pyplot as plt
import seaborn as sbs
from random import randint

def load_from_txt(filelocation):
    width_re = re.compile('\d*(?=,)')
    height_re = re.compile('\d*(?=\))')
    image_sizes_as_tuples = []

    with open(filelocation, 'r') as fp:
        raw_image_sizes = fp.readlines()

    for raw_image_size in raw_image_sizes:
        width = width_re.search(raw_image_size).group(0)
        height = height_re.search(raw_image_size).group(0)
        image_sizes_as_tuples.append((int(width), int(height)))

    return image_sizes_as_tuples

if __name__ == '__main__':
    img_sizes = load_from_txt('img_res.txt')

    img_sizes_prod = [img[0]*img[1] for img in img_sizes]
    img_size_ratios = [img[0]/img[1] for img in img_sizes]

    sbs.set_theme()

    prod_label = 'product of the image sizes'
    img_size_prod_d = {prod_label: img_sizes_prod}
    sbs.kdeplot(data=img_size_prod_d, x=prod_label)
    print("plotting image size ratios")
    plt.show()

    ratios_label = 'image ratios'
    img_size_ratios_d = {ratios_label: img_size_ratios}
    sbs.kdeplot(data=img_size_ratios_d, x=ratios_label)
    print("plotting product of image sizes")
    plt.show()

    #plotting 100k random integers from [1;10], which should be uniform
    r_numbers = [randint(1,10) for i in range(100000)]
    r_numbers_label = '100k random numbers [1;10]'
    r_numbers_d = {r_numbers_label: r_numbers}
    sbs.kdeplot(data=r_numbers_d, x=r_numbers_label)
    print("plotting 100k random integers from [1;10], which should be uniform")
    plt.show()
