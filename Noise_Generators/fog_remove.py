import cv2
import math
import numpy as np
from PIL import Image
import sys
#Greatly inspired by https://github.com/He-Zhang/image_dehaze
class fog_remove():
    kernel_size = 15

    def __init__(self,config):
        self.Keys=['kernel']
        for key in self.Keys:
            if key in config:
                setattr(self, key, config.get(key))

    def DarkChannel(self,im,sz):
        b,g,r = cv2.split(im)
        dc = cv2.min(cv2.min(r,g),b);
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
        dark = cv2.erode(dc,kernel)
        return dark

    def AtmLight(self,im,dark):
        [h,w] = im.shape[:2]
        imsz = h*w
        numpx = int(max(math.floor(imsz/1000),1))
        darkvec = dark.reshape(imsz,1);
        imvec = im.reshape(imsz,3);

        indices = darkvec.argsort();
        indices = indices[imsz-numpx::]

        atmsum = np.zeros([1,3])
        for ind in range(1,numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx;
        return A

    def TransmissionEstimate(self,im,A,sz):
        omega = 0.95;
        im3 = np.empty(im.shape,im.dtype);

        for ind in range(0,3):
            im3[:,:,ind] = im[:,:,ind]/A[0,ind]

        transmission = 1 - omega*self.DarkChannel(im3,sz);
        return transmission

    def Guidedfilter(self,im,p,r,eps):
        mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
        mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
        mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
        cov_Ip = mean_Ip - mean_I*mean_p;

        mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
        var_I   = mean_II - mean_I*mean_I;

        a = cov_Ip/(var_I + eps);
        b = mean_p - a*mean_I;

        mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
        mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

        q = mean_a*im + mean_b;
        return q;

    def TransmissionRefine(self,im,et):
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
        gray = np.float64(gray)/255;
        r = 60;
        eps = 0.0001;
        t = self.Guidedfilter(gray,et,r,eps);

        return t;

    def Recover(self,im,t,A,tx = 0.1):
        res = np.empty(im.shape,im.dtype);
        t = cv2.max(t,tx);

        for ind in range(0,3):
            res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

        return res

    def de_fog(self, img):
        src = np.array(img.convert('RGB')) 
        src = src[:, :, ::-1].copy()  

        I = src.astype('float64')/255;
    
        dark = self.DarkChannel(I,self.kernel_size);
        A = self.AtmLight(I,dark);
        te = self.TransmissionEstimate(I,A,self.kernel_size);
        t = self.TransmissionRefine(src,te);
        J = self.Recover(I,t,A,0.1);

        J = J*255
        J = cv2.cvtColor(J.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(J)
        return im_pil

    def __add__(self, img):
        return self.de_fog(img)
