from PIL import Image
import matplotlib.image as mpimg
import  numpy as np
import cv2

class homomorphic():

    # Convert image to a grayscaled image.
    def grayscale_image(self, img_path):
        img = Image.open(img_path).convert('LA')
        img.save('output/grayscaled.png')

    # Convert image to YCRCB color space 
    def convert_image_YCC(self, img_path): 
        imgog = Image.open(img_path)
        imgYCC = imgog.convert('YCbCr')
        bands = imgYCC.getbands()
        ycc_list = list(imgYCC.getdata())
        # print('ycc list' , ycc_list[0])
        ycc_list = np.reshape(imgYCC, (imgog.size[1], imgog.size[0], 3))
        ycc_list.astype(np.uint8)

        # print(self.y.shape)
        # print(self.cb.dtype)
        # print(self.cb.shape)
        # print(self.cr.dtype)
        # print(self.cr.shape)
        self.y = Image.fromarray(ycc_list[:,:,0], "L")
        self.cb = Image.fromarray(ycc_list[:,:,1], "L")
        self.cr = Image.fromarray(ycc_list[:,:,2], "L")

        # Image.fromarray(ycc_list[:,:,0], "L").save('output/Y.png')
        # Image.fromarray(ycc_list[:,:,1], "L").save('output/cb.png')
        # Image.fromarray(ycc_list[:,:,2], "L").save('output/cr.png')
        # imgres = cv2.cvtColor(np.float32(imgYCC), cv2.COLOR_BGR2YCR_CB)

    def gaussian_filter(self):
        P = self.y.shape[0]/2
        Q = self.y.shape[1]/2
        H = np.zeros(self.y.shape)
        U, V = np.meshgrid(range(self.y.shape[0]), range(self.y.shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*800**2)))
        print(H)
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
        self.y = (1.3 + 3*filter)*self.y

    def merge_image(self): 
        # temp = [arr1,arr2[1],arr2[2]]
        # for i in range(len(temp)):
        #     im = Image.fromarray(np.uint8(temp[i]))
        #     temp[i] = im.convert('L')
        # print(temp[0])
        self.y = Image.fromarray(self.y)
        # print(type(self.y))
        # print(type(self.cb), self.cb.size)
        # print(type(self.cr))
        y = self.y.convert('L')
        cb = self.cb.convert('L')
        cr = self.cr.convert('L')
        # y = y.convert('RGB')
        # cb = cb.convert('RGB')
        # cr = cr.convert('RGB')
        # y_arr = np.asarray(y)
        # cb_arr = np.asarray(cb)
        # cr_arr = np.asarray(cr)
        # print(y_arr.dtype,cb_arr.dtype,cr_arr.dtype)
        return Image.merge('YCbCr',(y, cb, cr)).convert('RGB')

if __name__=='__main__':
    homo = homomorphic()
    path = 'C:/Users/roni/Desktop/Project/homomorphic/src/output/Y.png'
    conv_img = homo.convert_image_YCC(path)
    freq_dom = homo.transform_into_freqdomain()
    ft = homo.fourier_transfor()
    gaus_filter = homo.gaussian_filter()
    filtered = homo.apply_highpass_filter(gaus_filter)
    invers = homo.inverse_fourier_transfor()
    exp = homo.exponential_func()
    # img = homo.merge_image()
    a = homo.merge_image()
    a.save('output/res.png')
    # b.show()
    # c.show()
    
    # print(homo.cr.shape)
