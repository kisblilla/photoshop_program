

from cv2 import transform
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import PIL
import timeit 
import matplotlib.pyplot as plt


def main():
    st.title("Image editor")
    st.text("Kisbenedek Lilla")
        
    enhance_type =st.sidebar.radio("Enhance Type", ["Negalas", "Gamma transzformacio", "Logaritmus transzformacio", "Szurkites","Hisztogram",
    "Hisztogram kiegyenlítés", "Átlagoló szűrő", "Gauss szűrő", "Sobel operator", "Laplace operator", "Harris sarok detektalas"])

    image_file=st.file_uploader("Kép feltöltése", type=['jpg', 'jpeg'])
      
    
    if image_file is not None:
        image_process(image_file, enhance_type)

    else:
        image_file2='example.jpg'
        if enhance_type == "Harris sarok detektalas":
            image_file2='Harrisexample.jpg'
        image_process(image_file2, enhance_type)

def image_process(image_file, enhance_type):
    start = timeit.default_timer()
    image_file=image_file
    enhance_type=enhance_type
    col1, col2 = st.columns( [0.5, 0.5])
    
    with col1:
        st.markdown('<p style="text-align: center;">Előtte</p>',unsafe_allow_html=True)
        st.image(image_file)
        st.text("Futási idő [s]:")
        

    with col2:
        st.markdown('<p style="text-align: center;">Utána</p>',unsafe_allow_html=True)

        if enhance_type == 'Negalas':
            image_array=np.array(Image.open(image_file))
            image_array2 = 255 - image_array
            img = PIL.Image.fromarray(image_array2)
            st.image(img)
            stop = timeit.default_timer()
            st.text(stop - start)

        if enhance_type == 'Gamma transzformacio':
            c_rate=st.sidebar.slider("Gamma",0.5, 3.5)
            image_array= np.array(Image.open(image_file))
            image_array2 = 255.0 * (image_array / 255.0)**c_rate
            img = Image.fromarray(np.uint8(image_array2))
            st.image(img)
            stop = timeit.default_timer()
            st.text(stop - start)

        if enhance_type == 'Logaritmus transzformacio':
            image_array= np.array(Image.open(image_file))
            image_array2 = 255 / np.log(1 + np.max(image_array))
            image_array3 = image_array2 * (np.log(image_array + 1))
            img = Image.fromarray(np.uint8(image_array3))
            st.image(img)
            stop = timeit.default_timer()
            st.text(stop - start)

        if enhance_type=='Szurkites':
            image=Image.open(image_file)
            image_array=np.array(image)
            #image_array=np.array(Image.open(image_file))
            for i in range(len(image_array)):
                for j in range(len(image_array[i])):
                    red = image_array[i, j, 2]
                    green = image_array[i, j, 1]
                    blue = image_array[i, j, 0]
                    grayscale_value = blue*0.114 + green*0.587 + red*0.299
                    image_array[i,j] = grayscale_value
            img = PIL.Image.fromarray(image_array)
            st.image(img)
            stop = timeit.default_timer()
            st.text(stop - start)

        if enhance_type=='Hisztogram':
            img = plt.imread(image_file)
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            
            plt.subplots_adjust(hspace=0.5, wspace=0.5)

            plt.subplot(2, 2, 1)
            plt.title('Red')
            hist, bins = np.histogram(r.ravel(), bins=256, range=(0, 255))
            plt.bar(bins[:-1], hist)

            plt.subplot(2, 2, 2)
            plt.title('Green')
            hist, bins = np.histogram(g.ravel(), bins=256, range=(0, 255))
            plt.bar(bins[:-1], hist)

            plt.subplot(2, 2, 3)
            plt.title('Blue')
            hist, bins = np.histogram(b.ravel(), bins=256, range=(0, 255))
            plt.bar(bins[:-1], hist)
            st.pyplot(plt)
            stop = timeit.default_timer()
            st.text(stop - start)

        if enhance_type=='Hisztogram kiegyenlítés':
            
            img_array= np.array(Image.open(image_file))
            hist_array = np.bincount(img_array.flatten(), minlength=256)
            sum_pixels = np.sum(hist_array)
            hist_array = hist_array/sum_pixels
            hist_array = np.cumsum(hist_array)

            floor_inp = np.floor(255 * hist_array).astype(np.uint8)

            img_list = list(img_array.flatten())

            eq_list = [floor_inp[p] for p in img_list]

            eq_array = np.reshape(np.asarray(eq_list), img_array.shape)
            eq_img = Image.fromarray(eq_array)
            st.image(eq_img)
            stop = timeit.default_timer()
            st.text(stop - start)
                            


        if enhance_type=='Átlagoló szűrő':

            c_rate=st.sidebar.slider("Átlagoló (n x n)",5, 36)
            def meanFilter(img,k):
                size = k // 2
                w,h,c = img.shape

                _img = np.zeros((w+2*size,h+2*size,c), dtype=np.uint8)
                _img[size:size+w,size:size+h] = img.copy().astype(np.uint8)
                dst = _img.copy()

                ker = np.zeros((k,k), dtype=float)
                for x in range(-1*size,k-size):
                    for y in range(-1*size,k-size):
                        ker[x+size,y+size] = (1/k**2)

                for x in range(w):
                    for y in range(h):
                        for z in range(c):
                            dst[x+size,y+size,z] = np.sum(ker*_img[x:x+k,y:y+k,z])

                dst = dst[size:size+w,size:size+h].astype(np.uint8)

                return dst

            img=np.array(Image.open(image_file))
            img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = meanFilter(img,c_rate)

            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            st.image(img)
            stop = timeit.default_timer()
            st.text(stop - start)

        if enhance_type=='Gauss szűrő':
            filter_size=st.sidebar.slider("Szűrő értéke",1, 50)
            standard_deviation=st.sidebar.slider("Szórás",1, 10)
            def gaussianFilter(img,k,s):
                w,h,c = img.shape
                size = k // 2

                _img = np.zeros((w+2*size,h+2*size,c), dtype=np.uint8)
                _img[size:size+w,size:size+h] = img.copy().astype(np.uint8)
                dst = _img.copy()

                ker = np.zeros((k,k), dtype=float)
                for x in range(-1*size,k-size):
                    for y in range(-1*size,k-size):
                        ker[x+size,y+size] = (1/(2*np.pi*(s**2)))*np.exp(-1*(x**2+y**2)/(2*(s**2)))
                ker /= ker.sum()

                for x in range(w):
                    for y in range(h):
                        for z in range(c):
                            dst[x+size,y+size,z] = np.sum(ker*_img[x:x+k,y:y+k,z])
                dst = dst[size:size+w,size:size+h].astype(np.uint8)
                return dst


            img=np.array(Image.open(image_file))
            img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = gaussianFilter(img,filter_size,standard_deviation)

            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            st.image(img)
            stop = timeit.default_timer()
            st.text(stop - start)


        if enhance_type=='Sobel operator':
            img = np.array(Image.open(image_file)).astype(np.uint8)
            gray_img = np.round(0.299 * img[:, :, 0] +
                                0.587 * img[:, :, 1] +
                                0.114 * img[:, :, 2]).astype(np.uint8)

            h, w = gray_img.shape
            horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  
            vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) 

            gradient = np.zeros((h, w))
            horizontal_image = np.zeros((h, w))
            vertical_image = np.zeros((h, w))

            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    grad_hor = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                                    (horizontal[0, 1] * gray_img[i - 1, j]) + \
                                    (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                                    (horizontal[1, 0] * gray_img[i, j - 1]) + \
                                    (horizontal[1, 1] * gray_img[i, j]) + \
                                    (horizontal[1, 2] * gray_img[i, j + 1]) + \
                                    (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                                    (horizontal[2, 1] * gray_img[i + 1, j]) + \
                                    (horizontal[2, 2] * gray_img[i + 1, j + 1])

                    horizontal_image[i - 1, j - 1] = abs(grad_hor)

                    grad_ver = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                                (vertical[0, 1] * gray_img[i - 1, j]) + \
                                (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                                (vertical[1, 0] * gray_img[i, j - 1]) + \
                                (vertical[1, 1] * gray_img[i, j]) + \
                                (vertical[1, 2] * gray_img[i, j + 1]) + \
                                (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                                (vertical[2, 1] * gray_img[i + 1, j]) + \
                                (vertical[2, 2] * gray_img[i + 1, j + 1])

                    vertical_image[i - 1, j - 1] = abs(grad_ver)

                    edgem = np.sqrt(pow(grad_hor, 2.0) + pow(grad_ver, 2.0))
                    gradient[i - 1, j - 1] = edgem
                    
            img = Image.fromarray(gradient.astype(np.uint8))
            st.image(img)
            stop = timeit.default_timer()
            st.text(stop - start)

        if enhance_type=='Laplace operator':

            def Laplacian(w, h):

                filter = np.zeros((w, h, 2), dtype=np.float32)
            
                for i in range(w):
                    for j in range(h):
                        filter[i][j] = -((i-w/2)**2 + (j-h/2)**2)

                return filter

            img = plt.imread(image_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = img.shape

            ft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
            ft_shifted = np.fft.fftshift(ft)

            filter = Laplacian(h, w)
            filterMag = 20 * np.log(cv2.magnitude(filter[:, :, 0], filter[:, :, 1]))

            applied = ft_shifted * filter
            f_ishift = np.fft.ifftshift(applied)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            plt.axis('off')
            plt.figure(figsize=(10, 10))    
            plt.axis('off')
            plt.imshow(img_back, cmap="gray")
            st.pyplot(plt)
            stop = timeit.default_timer()
            st.text(stop - start)

        if enhance_type=='Harris sarok detektalas':


            def conv(I, filter_x, filter_y):
                h, w = I.shape[:2]
                n = filter_x.shape[0]//2
                I_a = np.zeros(I.shape)
                I_b = np.zeros(I.shape)
                for x in range(n, w-n):
                    patch = I[:, x-n:x+n+1]
                    I_a[:, x] = np.sum(patch * filter_x, 1)
                filter_y = np.expand_dims(filter_y, 1)
                for y in range(n, h-n):
                    patch = I_a[y-n:y+n+1, :]
                    I_b[y, :] = np.sum(patch * filter_y, 0)
                return I_b

            def gaussian(n, sigma=None):
                if sigma is None:
                    sigma = 0.3 * (n // 2) + 0.8
                    X = np.arange(-(n//2), n//2+1)
                    kernel = np.exp(-(X**2)/(2*sigma**2))
                    return kernel


            def detect(I, n_g=5, n_w=5, k=0.06):
                h, w = I.shape
                sobel_1 = np.array([-1, 0, 1])
                sobel_2 = np.array([1, 2, 1])
                I_x = conv(I, sobel_1, sobel_2)
                I_y = conv(I, sobel_2, sobel_1)
                g_kernel = gaussian(n_g)
                I_x = conv(I_x, g_kernel, g_kernel)
                I_y = conv(I_y, g_kernel, g_kernel)
                D_temp = np.zeros((h, w, 2, 2))
                D_temp[:, :, 0, 0] = np.square(I_x)
                D_temp[:, :, 0, 1] = I_x * I_y
                D_temp[:, :, 1, 0] = D_temp[:, :, 0, 1]
                D_temp[:, :, 1, 1] = np.square(I_y)
                g_filter = gaussian(n_w)
                g_filter = np.dstack([g_filter] * 4).reshape(n_w, 2, 2)
                D = conv(D_temp, g_filter, g_filter)
                P = D[:, :, 0, 0]
                Q = D[:, :, 0, 1]
                R = D[:, :, 1, 1]
                T1 = (P + R) / 2
                T2 = np.sqrt(np.square(P - R) + 4 * np.square(Q)) / 2
                L_1 = T1 - T2
                L_2 = T1 + T2
                C = L_1 * L_2 - k * np.square(L_1 + L_2)
                return C, I_x, I_y, L_1, L_2

        
            img_path = image_file
            img = np.array(Image.open(img_path).convert('L'))
            img = (img - img.min())/(img.max()-img.min())
            C, I_x, I_y, L_1, L_2 = detect(img, k=0.06)
            C = (C - C.min())/(C.max()-C.min())


            plt.figure(figsize=(13, 5))

            plt.subplot(121)
            plt.title('$I_x$')
            plt.imshow(I_x, cmap='gray')
            plt.subplot(122)
            plt.title('$I_y$')
            plt.imshow(I_y, cmap='gray')
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)


            plt.figure(figsize=(13, 5))
            plt.subplot(121)
            plt.imshow(L_1, cmap='gnuplot')
            # plt.colorbar()
            plt.subplot(122)
            plt.imshow(L_2, cmap='gnuplot')
            # plt.colorbar()
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)


            plt.figure(figsize=(13, 5))
            plt.subplot(121)
            plt.imshow(C-0.457, cmap='gnuplot')
            plt.subplot(122)
            plt.imshow(img/2+2*C*(C >= 0.457), cmap='gnuplot')
            plt.title('Detektált sarkok')
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
            stop = timeit.default_timer()
            st.text(stop - start)




if __name__=='__main__':

    main()