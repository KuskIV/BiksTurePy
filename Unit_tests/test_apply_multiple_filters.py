import unittest
import os,sys,inspect
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from Noise_Generators.noise_main import apply_multiple_filters, premade_single_filter

class Test_apply_multiple_filters(unittest.TestCase):
    #creates a list of a 100 images

    def test_apply_multiple_filters_nochunk_equal_split_even(self):

        img_list = []
        lables_list = []
        Total_entries = 100
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=0)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])

        self.assertEqual(snow,rain)

    def test_apply_multiple_filters_nochunk_right_priority_split_uneven(self):

        img_list = []
        lables_list = []
        Total_entries = 101
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=0)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])

        self.assertEqual(snow+1,rain)

    def test_apply_multiple_filters_nochunk_sum_to_total_even(self):

        img_list = []
        lables_list = []
        Total_entries = 100
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=0)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])
        result = snow + rain

        self.assertEqual(Total_entries,result)

    def test_apply_multiple_filters_nochunk_sum_to_total_uneven(self):

        img_list = []
        lables_list = []
        Total_entries = 101
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=0)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])
        result = snow + rain

        self.assertEqual(Total_entries,result)

    def test_apply_multiple_filters_nochunk_sum_to_total_3_even(self):

        img_list = []
        lables_list = []
        Total_entries = 100
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain"),"night":premade_single_filter("night")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=0)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])
        night = sum([1 if "night" == x else 0 for x in noises])
        result = snow + rain + night

        self.assertEqual(Total_entries,result)

    def test_apply_multiple_filters_nochunk_sum_to_total_3_uneven(self):

        img_list = []
        lables_list = []
        Total_entries = 101
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain"),"night":premade_single_filter("night")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=0)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])
        night = sum([1 if "night" == x else 0 for x in noises])
        result = snow + rain + night

        self.assertEqual(Total_entries,result)

    def test_apply_multiple_filters_nochunk_right_prioty_3_even(self):

        img_list = []
        lables_list = []
        Total_entries = 100
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain"),"night":premade_single_filter("night")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=0)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])
        night = sum([1 if "night" == x else 0 for x in noises])

        self.assertEqual(sum([snow,rain])/2,night-1)

    def test_apply_multiple_filters_chunk_sum_to_total_even(self):

        img_list = []
        lables_list = []
        Total_entries = 100
        chungus = 3
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=chungus)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])
        result = snow + rain
        expected = (Total_entries/(chungus+len(filters)))*len(filters)
        self.assertEqual(int(expected),result)

    def test_apply_multiple_filters_chunk_sum_to_total_uneven(self):

        img_list = []
        lables_list = []
        Total_entries = 101
        chungus=3
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=chungus)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])
        result = snow + rain
        expected = (Total_entries/(chungus+len(filters)))*len(filters)
        self.assertEqual(int(expected),result)

    def test_apply_multiple_filters_chunk_sum_to_total_3_uneven(self):
        img_list = []
        lables_list = []
        Total_entries = 101
        chungus=3
        i = 0
        for i in range(Total_entries):
            img_list.append(Image.open("Unit_tests/Tiny_test_img/00002_00000.ppm"))
            lables_list.append("yeet")

        filters = {"snow":premade_single_filter("snow"),"rain":premade_single_filter("rain"),"night":premade_single_filter("night")}
        imgs = apply_multiple_filters((img_list,lables_list),mode="linear",KeepOriginal=False,filters=filters, chungus=chungus)
        noises = [item[1] for item in imgs]
        snow = sum([1 if "snow" == x else 0 for x in noises])
        rain = sum([1 if "rain" == x else 0 for x in noises])
        night = sum([1 if "night" == x else 0 for x in noises])
        result = snow + rain + night
        expected = (Total_entries/(chungus+len(filters)))*len(filters)
        self.assertEqual(int(expected),result)