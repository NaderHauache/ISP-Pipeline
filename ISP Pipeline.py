#Libraries
import rawpy
import numpy as np
import imageio.v2 as io
import imquality.brisque as brisque
from dom import DOM
from colour_demosaicing import (
    demosaicing_CFA_Bayer_Malvar2004
)

#Path and Name Definitions
path = "RAW_Images/"
name_pic = "dog"
ext = ".ARW"
saveOp = False
EvalOp = True
bit8 = 255
bit16 = 65535


#Raw Object from RawPy acquisition - Includes Metadata 
raw = rawpy.imread(path+name_pic+ext)
n_colors = raw.num_colors #param which says how much colors CFA filter have

def getIndexPattern():
    #getting type colors and CFA pattern
    colors = np.frombuffer(raw.color_desc, dtype=np.byte)
    pattern = np.array(raw.raw_pattern)

    # Appoints correspontend colors on pattern to understand where is each color on bayer pattern
    idx0 = np.where(colors[pattern] == colors[0])
    idx1 = np.where(colors[pattern] == colors[1])
    idx2 = np.where(colors[pattern] == colors[2])

    return [idx0, idx1, idx2]

def readImage(name_pic):
    #Read and Save an Pure Raw Image
    imageArray = np.array(raw.raw_image, dtype=np.double)

    print("Loading Image...")

    if (saveOp):
        
        io.imwrite("Output/"+name_pic+"_pure"+".tiff", imageArray.astype(np.uint16), format="tiff")    


    return imageArray

def linearizationOp(imageArray):

    #Creating an black level array with the same shape of original image 
    black_mask = np.reshape(raw.black_level_per_channel, (2,2))
    black_mask = np.tile(black_mask,(imageArray.shape[0]//2,imageArray.shape[1]//2))

    #Formula to Black Level Correction: (raw_pixel - black_level) / (max_value - black_level)
    #All the values are normalized to interval between 0 ~ 1 to avoid magenta cast
    #imageArray = (imageArray - black_mask) / (raw.white_level - black_mask[0][0])
    imageArray = (imageArray - black_mask) / (raw.white_level - black_mask)

    print("Image Black Level Correction: DONE")

    if (saveOp):
        
        io.imwrite("Output/"+name_pic+"_linearized"+".tiff", (imageArray*bit16).astype(np.uint16), format="tiff")

    
    return imageArray


def wbOp(imageArray, idx):

    #Getting info of CFA Pattern and White Balance Values provided by metadata
    wb_correctionTable = raw.camera_whitebalance

    # Perfoming the White Balance Operation
    imageRaw_wb = np.zeros((2, 2), dtype=np.double) 
    imageRaw_wb[idx[0]] = wb_correctionTable[0] / wb_correctionTable[1]
    imageRaw_wb[idx[1]] = wb_correctionTable[1] / wb_correctionTable[1]
    imageRaw_wb[idx[2]] = wb_correctionTable[2] / wb_correctionTable[1]


    imageRaw_wb = np.tile(imageRaw_wb, (imageArray.shape[0]//2, imageArray.shape[1]//2))

    #clip function on Numpy maps the value into the interval of [0 ~ 1]. It's done to avoid magenta cast
    imageRaw_wb = np.clip(imageArray * imageRaw_wb, 0, 1) 

    print("White Balance Operation: DONE")

    if (saveOp):
        io.imwrite("Output/"+name_pic+"_WB"+".tiff", (imageRaw_wb*bit16).astype(np.uint16), format="tiff")

    
    return imageRaw_wb 

def saveBayerImage(imageArray,idx):

    print("[OPTIONAL]CFA Representation Process: Running... (Can take a little while)")

    R_channel = np.zeros((2, 2), dtype=np.double)
    R_channel[idx[0]] = 1
    R_channel = np.tile(R_channel, (imageArray.shape[0]//2, imageArray.shape[1]//2))
    R_channel = imageArray * R_channel *bit8

    G_channel = np.zeros((2, 2), dtype=np.double)
    G_channel[idx[1]] = 1
    G_channel = np.tile(G_channel, (imageArray.shape[0]//2, imageArray.shape[1]//2))
    G_channel = imageArray * G_channel *bit8

    B_channel = np.zeros((2, 2), dtype=np.double)
    B_channel[idx[2]] = 1
    B_channel = np.tile(B_channel, (imageArray.shape[0]//2, imageArray.shape[1]//2))
    B_channel = imageArray * B_channel *bit8

    print("CFA Representation: DONE")
    io.imwrite("Output/"+name_pic+"_CFA"+".bmp", np.dstack((R_channel, G_channel, B_channel)).astype(np.uint8), format="BMP")  
    print("Saving Image 3channel...\n")

    
    return True

def demosaicing(imageArray):

    print("Demosaicing Process: Running... (Can take a little while)")
    imageArray = demosaicing_CFA_Bayer_Malvar2004(imageArray)
    
    print("Demosaic: DONE")

    if (saveOp):
        saveImage3channel(imageArray,"Demosaiced")
        #io.imwrite("Output/"+name_pic+"_Demosaiced"+".png", ((imageArray*bit8).astype(np.uint8)), format=".png")

    
    return imageArray

def colorSpaceConversion(imageArray):

    #Acquiring RGB to XYZ color Space
    XYZ_arrayCamera = np.array(raw.rgb_xyz_matrix[0:n_colors:], dtype = np.double)

    sRGBfromXYZConverter = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
    ], dtype= np.double)

    #Finding conversion matrix from XYZ to SRGB provided by camera (multiplies and normalize)
    sRGB_XYZConverter = np.dot(XYZ_arrayCamera, sRGBfromXYZConverter)
    norm = np.tile(np.sum(sRGB_XYZConverter,1),(3,1)).transpose()
    sRGB_XYZConverter = sRGB_XYZConverter/norm

    #Findings the inverse of matrix to find sRGB Values and compute the conversion using einsten sum
    sRGB = np.linalg.inv(sRGB_XYZConverter)

    sRGB = np.einsum('ij,...j', sRGB, imageArray)  # performs the matrix-vector product for each pixel

    print("Color Space Conversion(XYZ to sRGB): DONE")
     
    if (saveOp):
        saveImage3channel(sRGB,"sRGB-notGamma")
        #io.imwrite("Output/"+name_pic+"_sRGB-notGamma"+".png", ((sRGB*bit8).astype(np.uint8)), format=".png")


    return sRGB

def gammaCorrection(imageArray):

    i = imageArray < 0.0031308
    j = np.logical_not(i)
    imageArray[i] = 323 / 25 * imageArray[i]
    imageArray[j] = 211 / 200 * imageArray[j] ** (5 / 12) - 11 / 200
    imageArray = np.clip(imageArray, 0, 1)

    print("Gamma Correction Process: DONE")
    return imageArray

def saveImage3channel(imageArray,finalName):

    print("Saving Image 3channel...")
    io.imwrite("Output/"+name_pic+"_"+finalName+".bmp", ((imageArray*bit8).astype(np.uint8)), format="BMP")
    
    return True

indexes = getIndexPattern()
print("##### ISP Pipeline by Nader Hauache #####\n")
imageArray = readImage(name_pic)
imageArray = linearizationOp(imageArray)
imageArray = wbOp(imageArray, indexes)
if (saveOp):
    saveBayerImage(imageArray, indexes)
imageArray = demosaicing(imageArray)
imageArray = colorSpaceConversion(imageArray)
imageArray = gammaCorrection(imageArray)

if(saveImage3channel(imageArray,"sRGB")):
    print("\nISP Pipeline Process: COMPLETED!\n")

if(EvalOp):
    print("Evaluating IQ Metrics.. (usually takes about a minute... Just wait a little bit...)\n")

    #Explained on: https://github.com/ocampor/image-quality
    brisqueScore = brisque.score(imageArray)
    print("Brisque Score: " + str(brisqueScore))
    print("[Value Range: 0 ~ 100]\n")

    #Explained on: https://github.com/umang-singhal/pydom 
    iqs = DOM()
    sharp = iqs.get_sharpness("Output/"+name_pic+"_sRGB"+".bmp")
    print("Sharpness Estimation Score: " + str(sharp))
    print("[Value Range: 0 ~ sqrt(2)]\n")

