from PIL import Image
import matplotlib.image as mpimg
import  numpy as np
import cv2

class homomorphic():

    a = 0.2
    b = 1
    cut_off = 1
    # fil_order = 2

    def __init__(self,config):
        Keys=['a','b','cutoff']
        if Keys[0] in config:
            self.a = config.get(Keys[0])
        if config.get(Keys[1]) != None:
            self.b = config.get(Keys[1])
        if config.get(Keys[2]) != None:
            self.cut_off = config.get(Keys[2])
        # if config.get(Keys[3]) != None:
        #     self.fil_order = config.get(Keys[3])

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

    def gaussian_filter(self):
        P = self.y.shape[0]/2
        Q = self.y.shape[1]/2
        H = np.zeros(self.y.shape)
        U, V = np.meshgrid(range(self.y.shape[0]), range(self.y.shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*self.cut_off**2)))
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
        self.y = (self.a + self.b*filter)*self.y

    def merge_image(self): 
        self.y = Image.fromarray(self.y)
        y = self.y.convert('L')
        cb = self.cb.convert('L')
        cr = self.cr.convert('L')
        return Image.merge('YCbCr',(y, cb, cr)).convert('RGB')

    def homofy(self,img):
        conv_img = self.convert_image_YCC(img)
        freq_dom = self.transform_into_freqdomain()
        ft = self.fourier_transfor()
        gaus_filter = self.gaussian_filter()
        filtered = self.apply_highpass_filter(gaus_filter)
        invers = self.inverse_fourier_transfor()
        exp = self.exponential_func()
        return self.merge_image()

if __name__=='__main__':
    config = {'a':1,'b':0.5,'cutoff':3}
    homo = homomorphic(config)
    path = 'C:/Users/jeppe/Desktop/Homomorphic/2.png'
    img = Image.open(path)
    homo.homofy(img).save('C:/Users/jeppe/Desktop/Coroni_wrong/res.png')


