import logging
from PIL import Image 
import numpy as np

# Homomorphic filter class
class HomomorphicFilter:
    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H = None):

        if len(I.shape) != 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) != 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)
# End of class HomomorphicFilter

if __name__ == "__main__":
    import cv2
    filtred_bands = []
    temp = []
    im = Image.open(r"C:/Users/jeppe/Desktop/Homomorphic/2.png").convert('RGB')
    arr = np.array(im, dtype=float)

    path_in = 'C:/Users/jeppe/Desktop/Homomorphic'
    path_out = 'C:/Users/jeppe/Desktop/Homomorphic'
    img_path = '/2.png'
    for c in range(3):
        temp.append(arr[:,:,c])
    img_path_in = path_in + img_path
    img_path_out = path_out + '/filtered.png'

    # Main code
    for img in temp:
        homo_filter = HomomorphicFilter(a = 1.0, b = 1.5)
        filtred_bands.append(homo_filter.filter(I=img, filter_params=[30,2],filter = 'gaussian'))
    for i in range(len(filtred_bands)):
        im = Image.fromarray(np.uint8(filtred_bands[i]))
        filtred_bands[i] = im.convert('L')

    out = Image.merge("RGB", (filtred_bands[0], filtred_bands[1], filtred_bands[2]))
    out.save(img_path_out)