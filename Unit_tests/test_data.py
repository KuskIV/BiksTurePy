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

    def test_get_labels(self):
        img_path, labels = e.get_dataset_placements('Unit_tests\\Test_dataset')
        result = d.get_labels(img_path).tolist()
        res = [0,0,0,0]
        comp = result == res
        self.assertTrue(comp)

             
    def test_auto_reshape_image(self):
                #preparing result images
        flag = True
        result = []
        img_path, labels = e.get_dataset_placements('Unit_tests\\Test_dataset')
        imgs =[]
        for image in img_path:
            im_ppm = Image.open(image[0]) # Open as PIL image
            im_array = np.asarray(im_ppm) # Convert to numpy array
            imgs.append(im_array)
        Converted_result_imgs = d.auto_reshape_images((10,10),imgs)
        dtf = pd.DataFrame({'test':[Converted_result_imgs]})
        result.append(dtf.to_numpy())
        #result = pd.DataFrame({'test':Converted_result_imgs})

        #preparing refrence images
        ref_images =[]
        e.fileExstension = '.csv'
        img_path, labels = e.get_dataset_placements('Unit_tests\\test_dataset_normalized')
        e.fileExstension = '.ppm'
        for img in img_path:
            ref_images.append(pd.read_pickle(img[0]).to_numpy())
        res= result[0][0][0][0:4]
        res =res.tolist()
        ref =ref_images
        for i in range(len(ref)):
            ref[i]= ref[i][0][0][0:10].tolist()
            res[i]= res[i]
        for i in range(len(res)):
            comp = res[i]==ref[i]
            if(not comp):
                flag = False
                break
        self.assertTrue(flag)        
    

    def test_convert_imgs_to_numpy_arrays(self):
        #preparing result images
        flag = True
        result = []
        img_path, labels = e.get_dataset_placements('Unit_tests\\Test_dataset')
        Converted_result_imgs = d.convert_imgs_to_numpy_arrays(img_path)
        result.append(pd.DataFrame({'test':Converted_result_imgs}).to_numpy())
        #result = pd.DataFrame({'test':Converted_result_imgs})

        #preparing refrence images
        ref_images =[]
        e.fileExstension = '.csv'
        img_path, labels = e.get_dataset_placements('Unit_tests\\test_dataset_resized')
        e.fileExstension = '.ppm'
        for img in img_path:
            ref_images.append(pd.read_pickle(img[0]).to_numpy())
        res= result[0:4][0]
        res =res.tolist()
        ref =ref_images
        for i in range(len(ref)):
            ref[i]= ref[i][0][0][0]
            res[i]= res[i][0][0]
        for i in range(len(res)):
            comp = res[i]==ref[i]
            if(not comp.all()):
                flag = False
                break
        self.assertTrue(flag)        
         
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

"""
code to generate resized pictures as csv files
        img_path, amount = e.get_dataset_placements('Unit_tests\\Test_dataset')
        imgs =[]
        for image in img_path:
            im_ppm = Image.open(image[0]) # Open as PIL image
            im_array = np.asarray(im_ppm) # Convert to numpy array
            imgs.append(im_array)
        imgs = d.auto_reshape_images((10,10),imgs)
        i = 0
        for img in imgs:
            df = pd.DataFrame({"test": [img]})
            df.to_pickle('Unit_tests\\test_dataset_normalized\\00\\0000'+str(i)+'.csv')
            #np.savetxt('Unit_tests\\test_dataset_resized\\00\\0000'+str(i)+'.csv',img,delimiter=',')
            i+=1 
"""