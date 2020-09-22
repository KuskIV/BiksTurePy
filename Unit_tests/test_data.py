import unittest
import os,sys,inspect
import  numpy as np
from PIL import Image
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import data as d
import extract as e

class test_data(unittest.TestCase):

    #def test_get_data(self):
         

    def test_convert_imgs_to_numpy_arrays(self):
        #preparing result images
        result = []
        img_path, amount = e.get_dataset_placements('Unit_tests\\Test_dataset')
        Converted_result_imgs = d.convert_imgs_to_numpy_arrays(img_path)
        for imgs in Converted_result_imgs:
            result.append(pd.DataFrame({'test':Converted_result_imgs}).to_numpy())
        #result = pd.DataFrame({'test':Converted_result_imgs})

        #preparing refrence images
        ref_images =[]
        e.fileExstension = '.csv'
        img_path, amount = e.get_dataset_placements('Unit_tests\\test_dataset_resized')
        e.fileExstension = '.ppm'
        for img in img_path:
            ref_images.append(pd.read_pickle(img[0]).to_numpy())
        print(result[0][0])
        print(ref_images[0][0])
        self.assertTrue(result[0][0]==ref_images[0][])        
         
if __name__ == '__main__':
    unittest.main()

"""
code for generating the initial test images. for the test_convert_imgs_to_numpy_arrays.
        img_path, amount = e.get_dataset_placements('Unit_tests\\Test_dataset')
        imgs = d.convert_imgs_to_numpy_arrays(img_path)
        i = 0
        for img in imgs:
            #image = Image.open(img)
            image = np.array(img)
            im = Image.fromarray(image.astype(np.uint8))
            im.save('Unit_tests\\test_dataset_resized\\0000'+str(i)+'.png')
            i+=1   
"""

"""
Code to add the 3d array to a panda frame and save it.
        img_path, amount = e.get_dataset_placements('Unit_tests\\Test_dataset')
        imgs = d.convert_imgs_to_numpy_arrays(img_path)
        i = 0
        for img in imgs:
            df = pd.DataFrame({"test": [img]})
            df.to_pickle('Unit_tests\\test_dataset_resized\\00\\0000'+str(i)+'.csv')
            #np.savetxt('Unit_tests\\test_dataset_resized\\00\\0000'+str(i)+'.csv',img,delimiter=',')
            i+=1  
"""