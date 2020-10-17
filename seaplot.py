import re
import matplotlib.pyplot as plt
import seaborn as sbs

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

    sbs.kdeplot(data=img_size_ratios)
    print("plotting image size ratios")
    plt.show()

    sbs.kdeplot(data=img_sizes_prod)
    print("plotting product of image sizes")
    plt.show()
