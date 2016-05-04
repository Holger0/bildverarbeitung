import numpy as np
import matplotlib.pyplot as plt
import pylab
plt.rcParams['figure.figsize'] = (12, 12)   # This makes the plot bigger
from skimage.data import astronaut
from skimage.color import rgb2gray
# We use a gray image. All the algorithms should work with color images too.
img = rgb2gray(astronaut() / 255.)
#plt.imshow(img, cmap='gray')
#plt.show()
def derive_y(image):
    """Computes the derivative of the image w.r.t the y coordinate"""
    derived_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if y + 1 < image.shape[1] and y - 1 > 0:
                derived_image[x,y] = image[x, y - 1] - image[x, y + 1]
    return derived_image

def derive_x(image):
    """Computes the derivative of the image w.r.t the x coordinate"""
    derived_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if x + 1 < image.shape[1] and x - 1 > 0:
                derived_image[x,y] = image[x - 1, y] - image[x + 1, y]
    return derived_image
#dx_img = derive_x(img)
#dy_img = derive_y(img)
#plt.figure(figsize=(18, 12))
#plt.subplot(131)
#plt.imshow(img, cmap='gray')
#plt.subplot(132)
#plt.imshow(dx_img, cmap='gray')
#plt.subplot(133)
#plt.imshow(dy_img, cmap='gray')
#plt.show()
T_scale = np.array([
    [0.75, 0, 0],
    [0, 0.75, 0],
    [0, 0, 1],
])
T_affine = np.array([
    [0.75, 0.2, 100],
    [-0.2, 0.75, 100],
    [0, 0.001, 1],
])
def affine_transformation(img, matrix):
    # your code here
    #Dimension des neuen Bildes Berechnen, die maximalen Werte für x und y sind in den Ecken vorhanden
    maxX=0
    maxY=0
    #alle Ecken durchgehen
    #Position im transformierten Bild berechnen und maximum suchen
    for i in [0,1]:
        for j in [0,1]:
            aktPos=matrix @ [i*len(img),j*len(img[0]),1]
            if maxX<aktPos[0]:
                maxX=aktPos[0]
            if maxY<aktPos[1]:
                maxY=aktPos[1]
    imgResult=np.ones((int(maxX),int(maxY)))
    indicies = np.indices(imgResult.shape).reshape(2, -1)
    indicies_hg = np.concatenate([indicies, np.ones((1, indicies.shape[1]))], axis=0)
    #return indicies_hg
    values=bicubic_interpolation(img,np.linalg.inv(matrix) @ indicies_hg)
    for i in range(imgResult.shape[0]):
        for j in range(imgResult.shape[1]):
            imgResult[i,j] = values[i*(imgResult.shape[1])+j]
    return imgResult
def bicubic_interpolation(img, indicies):
    dx_img = derive_x(img)
    dy_img = derive_y(img)
    dxy_img = derive_x(dy_img)
    m =np.array([
        [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-3, 3, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2,-2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0],
        [-3, 0, 3, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0,-2, 0,-1, 0],
        [ 9,-9,-9, 9, 6, 3,-6,-3, 6,-6, 3,-3, 4, 2, 2, 1],
        [-6, 6, 6,-6,-3,-3, 3, 3,-4, 4,-2, 2,-2,-2,-1,-1],
        [ 2, 0,-2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [-6, 6, 6,-6,-4,-2, 4, 2,-3, 3,-3, 3,-2,-1,-2,-1],
        [ 4,-4,-4, 4, 2, 2,-2,-2, 2,-2, 2,-2, 1, 1, 1, 1],])
    result=np.ones(len(indicies[1]))
    for i in range(len(result)):
        aktIDX=indicies[:,i]
        AktX=int(aktIDX[0])
        AktY=int(aktIDX[1])
        if AktX<0 or AktY<0 or AktX>=img.shape[0] or AktY>=img.shape[1]:
            result[i]=0
        else:
            AktX=int(aktIDX[0])
            AktY=int(aktIDX[1])

            #Nachfolger bestimmen (ganz links existert keiner)
            NachfolgerX=AktX+1
            NachfolgerY=AktY+1
            if NachfolgerX==img.shape[0]:
                NachfolgerX=NachfolgerX-1
            if NachfolgerY==img.shape[1]:
                NachfolgerY=NachfolgerY-1
            DiffX=aktIDX[0]-AktX
            DiffY=aktIDX[1]-AktY
            #Berechnen der Werte a00...a33
            f=np.ones((16))
            f[0]=img[AktX, AktY]
            f[1]=img[NachfolgerX, AktY]
            f[2]=img[AktX, NachfolgerY]
            f[3]=img[NachfolgerX, NachfolgerY]

            f[4]=dx_img[AktX, AktY]
            f[5]=dx_img[NachfolgerX, AktY]
            f[6]=dx_img[AktX, NachfolgerY]
            f[7]=dx_img[NachfolgerX, NachfolgerY]

            f[8]=dy_img[AktX, AktY]
            f[9]=dy_img[NachfolgerX, AktY]
            f[10]=dy_img[AktX, NachfolgerY]
            f[11]=dy_img[NachfolgerX, NachfolgerY]

            f[12]=dxy_img[AktX, AktY]
            f[13]=dxy_img[NachfolgerX, AktY]
            f[14]=dxy_img[AktX, NachfolgerY]
            f[15]=dxy_img[NachfolgerX, NachfolgerY]

            
            
            a=m @ f
            #Reihenfolge in der Liste: a00, a10, a20, a30, a01, a11, a21, a31, a02, a12, a22, a32, a03, a13, a23, a33 -> aij=a[j*4+i]
            #Wert berechnen
            sum=0
            for j in range(4):
                for k in range(4):
                    sum=sum+a[k*4+j]*(DiffX**j)*(DiffY**k)
            result[i]=sum
    return result
img_scale = affine_transformation(img, T_scale)
img_affine = affine_transformation(img, T_affine)
plt.imshow(img_scale, cmap='gray')
plt.show()
plt.imshow(img_affine, cmap='gray')
plt.show()
