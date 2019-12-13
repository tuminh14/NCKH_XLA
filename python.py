import matlab.engine
import cv2
import matlab
import numpy as np
import os
import glob
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from timeit import default_timer as timer 


# X = []
def edge_detect(sequence_folder):
    i = 0
    eng = matlab.engine.start_matlab()
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    csv_final = []
    csv_final.append(['File name','psnr','num edge origin with sobel', 'num edge origin with canny' , 'num edge with ALM and saliency','num edge paper method'])
    for sq in sequence_folder:
        list_images_file = glob.glob(os.path.join(sq, '*.jpg'))
        for file_name in list_images_file:
            i +=1
            file_names = file_name.split('/')[-2]+ "/" + file_name.split('/')[-1] 
            ret = eng.Example_image_denoise(file_name,file_names)
            import_path = 'Deconvtv_out' +"/"+ file_names

            """Sobel"""
            origin = cv2.imread(file_name,0)
            sobelx_origin = cv2.Sobel(origin,cv2.CV_64F,1,0,ksize=3)  # x
            sobely_origin = cv2.Sobel(origin,cv2.CV_64F,0,1,ksize=3)  # y
            sobel_edge_origin = (sobelx_origin + sobely_origin)
            num_edge_origin = np.count_nonzero(sobel_edge_origin)
            sobel_edge_origin = (sobelx_origin + sobely_origin).astype('uint8')
            method = 'Sobel'
            export_path = '/media/tuminh14/New Volume/Window/NCKH/Egde_out' + '/' + method + '/' +file_names
            cv2.imwrite(export_path,sobel_edge_origin)
            """end"""

            """Canny"""
            edges_canny = cv2.Canny(origin,100,200)
            num_edge_canny = np.count_nonzero(edges_canny)
            method = 'Canny'
            export_path = '/media/tuminh14/New Volume/Window/NCKH/Egde_out' + '/' + method + '/' +file_names
            cv2.imwrite(export_path,edges_canny)
            """end"""

            """Sobel + Saliency + ALM"""
            deconvted = cv2.imread(import_path,0)
            (success, saliencyMap_deconvted) = saliency.computeSaliency(deconvted)
            saliencyMap_deconvted = (saliencyMap_deconvted*255).astype('uint8')
            sobelx_deconvted = cv2.Sobel(saliencyMap_deconvted,cv2.CV_64F,1,0,ksize=3)  # x
            sobely_deconvted = cv2.Sobel(saliencyMap_deconvted,cv2.CV_64F,0,1,ksize=3)  # y
            sobel_edge_deconvted = (sobelx_deconvted + sobely_deconvted)
            method = 'Sobel_saliency_alm'
            export_path = '/media/tuminh14/New Volume/Window/NCKH/Egde_out' + '/' + method + '/' +file_names
            cv2.imwrite(export_path,sobel_edge_deconvted)
            num_edge_deconvted = np.count_nonzero(sobel_edge_deconvted)
            psnr_out = psnr(origin,deconvted)
            """end"""

            """Paper recode"""
            method = "Paper method"
            image = cv2.imread(file_name,0)
            gau_image = cv2.GaussianBlur(image,(3,3),0)
            gau_image_canny = cv2.Canny(gau_image,100,200)
            num_edge_paper = np.count_nonzero(gau_image_canny)
            export_path = '/media/tuminh14/New Volume/Window/NCKH/Egde_out' + '/' + method + '/' + file_names
            cv2.imwrite(export_path, gau_image_canny)
            """end"""

            csv_sub_file = []
            csv_sub_file.append(file_names)
            csv_sub_file.append(str(psnr_out))
            csv_sub_file.append(str(num_edge_origin))
            csv_sub_file.append(str(num_edge_canny))
            csv_sub_file.append(str(num_edge_deconvted))
            csv_sub_file.append(str(num_edge_paper))
            csv_final.append(csv_sub_file)
    print (i)
    return csv_final

folder = 'Data/4saldetect/KNEE'
start = timer()
csv_out = pd.DataFrame(edge_detect(glob.glob(os.path.join(folder, '*/*'))))
print("time proccess= ",timer()-start)

csv_out.to_csv('psnr3.csv', index=False, header= False)

