from PIL import Image, ImageStat
import matplotlib.image as mpimg
import  numpy as np
import cv2
import math

class homomorphic():

    a = 0.5
    b = 0.5
    cut_off = 800
    # fil_order = 2

    def __init__(self,config):
        self.Keys=['a','b','cutoff']
        for key in self.Keys:
            if key in config:
                setattr(self, key, config.get(key))

    # Convert image to a grayscaled image.
    def grayscale_image(self, img_path):
        img = Image.open(img_path).convert('LA')
        img.save('output/grayscaled.png')

    # Convert image to YCRCB color space 
    def convert_image_YCC(self, imgog): 
        # imgog = Image.open(img_path)
        imgYCC = imgog.convert('YCbCr')
        bands = imgYCC.getbands()
        ycc_list = list(imgYCC.getdata())
        ycc_list = np.reshape(imgYCC, (imgog.size[1], imgog.size[0], 3))
        ycc_list.astype(np.uint8)

        self.y = Image.fromarray(ycc_list[:,:,0], "L")
        self.cb = Image.fromarray(ycc_list[:,:,1], "L")
        self.cr = Image.fromarray(ycc_list[:,:,2], "L")
        self.lumunosisty, self.max_intensity = self.find_intensity(self.y)  #add .copy if thunpnail is used

    def gaussian_filter(self):
        P = self.y.shape[0]/2
        Q = self.y.shape[1]/2
        H = np.zeros(self.y.shape)
        U, V = np.meshgrid(range(self.y.shape[0]), range(self.y.shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*self.cut_off**2)))
        return (1 - H)
    
    def butterworth_filter(self):
        P = self.y.shape[0]/2
        Q = self.y.shape[0]/2
        U, V = np.meshgrid(range(self.y.shape[0]), range(self.y.shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/self.cut_off**2)**2)
        return (1 - H)

    # Apply homophorbic filtering
    def transform_into_freqdomain(self):
        self.y = np.log1p(self.y)

    def fourier_transfor(self):
        self.y = np.fft.fft2(self.y)

    def inverse_fourier_transfor(self):
        self.y = np.fft.ifft2(self.y)
    
    def exponential_func(self):
        self.y = np.exp(np.real(self.y))-1

    def apply_highpass_filter(self, filter):
        # a = self.calc_scalar(self.lumunosisty)
        # if self.lumunosisty > 240: 
        #     self.b = 0.5
        self.y = (self.a + self.b*filter)*self.y
        # self.calc_scalar()
        # print ("b", self.b)
        # print ("a", self.a)

    def merge_image(self): 
        self.y = Image.fromarray(self.y)
        y = self.y.convert('L')
        cb = self.cb.convert('L')
        cr = self.cr.convert('L')
        return Image.merge('YCbCr',(y, cb, cr)).convert('RGB')

    def homofy(self,img):
        self.image_kernel(img)
        conv_img = self.convert_image_YCC(img)
        freq_dom = self.transform_into_freqdomain()
        ft = self.fourier_transfor()
        gaus_filter = self.gaussian_filter()
        butter_filter = self.butterworth_filter()
        filtered = self.apply_highpass_filter(butter_filter)
        invers = self.inverse_fourier_transfor()
        exp = self.exponential_func()
        return self.merge_image()

    def find_intensity(self,image): 
        # image.thumbnail((1,1))
        avg_pixels = image.getpixel((0,0))
        stat = ImageStat.Stat(self.y)
        mean= stat.mean[0]
        w,h = image.size
        w_c = int(w/2)
        h_c = int(h/2)
        h_p = int(h * 0.2)
        w_p = int(w * 0.2)
        # print ("w", w, "h", h)
        total = 0
        max_pix = 0
        smallest_pepe = 0
        kernal_size = h_p * w_p
        # print ("kernal size" , kernal_size)
        n = 0
        for i in range(0, w_p):
            for j in range(0, h_p):
                total += image.getpixel((i+w_c,j+h_c))
                if max_pix < image.getpixel((i+w_c,j+h_c)): 
                    max_pix = image.getpixel((i+w_c,j+h_c))
                if smallest_pepe > image.getpixel((i+w_c,j+h_c)): 
                    smallest_pepe = image.getpixel((i+w_c,j+h_c))
                n +=1
        mean_av = total / kernal_size
        # print ('a' , mean_av)
        # print ("total", total)
        # print ( "min pixel", smallest_pepe)
        # print ( "max pixel", max_pix)
        # # print ( "min pixel", smallest_pepe)
        # print ('mean' , mean_av)
        # percieved_light = math.sqrt(0.241 * (mean**2)) *2
        # rms = stat.rms[0]
        # avg = stat.average[0]
        scalar = self.calc_scalar(mean_av, max_pix)
        if mean_av >= 127:
            self.a = 1
            self.b = 0.5
        # elif max_pix > 240 and mean_av < 127 and mean_av <= 220: 
        #     self.a = 1.5
        #     self.b = 0.5
        elif mean_av <= 127 and mean_av >= 31:
            self.a = 1.1
            self.b = 0.5
        elif mean_av <= 70:
            self.a = 1.5
            self.b = 0.5

        else :
            print ("fuck")
        return mean_av, max_pix
        
    def calc_scalar(self, lumunosisty, intensity): 
        scalar = np.abs(lumunosisty - intensity)
        # print ('Scalar', scalar)
        # print ('Lumus', lumunosisty)
        norm = 1/(1 - np.exp(-scalar))
        # print ("norm", norm)
        return norm
    

        # print ('normalized', norm_ex)
    
    def ex_scalar(self,lumunostity):
        scalar = np.abs(lumunostity - 255)
        sigmoid = (1/1+np.exp(-scalar))
        inverse_sigmoid = ()
        return self
    
    def image_kernel(self, image) :
        w,h = image.size
        w_center = w/2
        h_center = h/2
        # print (w,h)
        # print (w_center, h_center)
    

if __name__=='__main__':
    config_normal= {'a':1,'b':0.5,'cutoff':800}
    config_dark = {'a':1.5,'b':0.5,'cutoff':800}
    config_lightaf = {'a':0.7,'b':0.3,'cutoff':30}
    config_config = {"normal": config_normal, "dark": config_dark, "light": config_lightaf}
    homo = homomorphic(config_dark)
    path2 = 'C:/Users/roni/Desktop/Project/BiksTurePy/Dataset/images/00011/00001_00024.ppm'
    path1 = 'C:/Users/roni/Desktop/Project/BiksTurePy/Dataset/images/00011/00010_00029.ppm'
    img_light = Image.open(path1)
    img_dark = Image.open(path2)
    print ('Dark Image')
    homo.homofy(img_dark).show()
    print ('Light Image')
    homo.homofy(img_light).show()
    
    # homo.homofy(img).save('C:/Users/jeppe/Desktop/Coroni_wrong/res.png')



