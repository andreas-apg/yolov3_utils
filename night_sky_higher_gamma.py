import sys
import numpy as np
import cv2
import argparse

#===============================================================================
# A: constroi o argumento e o argparser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type = str, required=True,
    help="caminho para o arquivo de entrada.")
ap.add_argument("-t", "--threshold", type = float, default = 0.5,
    help="limiar para filtrar a imagem de entrada pelo brilho em HLS.")
ap.add_argument("-w", "--weight", type = float, default = 0.1, 
    help="peso atribuido para a mascara de bloom em 0.0 < w <= 0.5.")
args = vars(ap.parse_args())

#===============================================================================
# A: constantes para acessar cada canal da imagem
BLUE = 0
GREEN = 1
RED = 2

HUE = 0
LUMINANCE = 1
SATURATION = 2


#-------------------------------------------------------------------------------
def luminance_threshold(img, threshold):
    '''
    Limiariza a imagem, separando apenas as partes mais claras de acordo com o canal LUMINANCE.
    Parâmetros:
        img: numpy array da imagem original em três canais RGB, de dimensões largura x altura x 3.
        threshold: float entre 0 e 1. Qualquer valor menor do que isso será considerado preto.
    Retorna:
        Uma segunda imagem de três canais HLS limiarizada tal que os pixels coloridos que sobraram
        são os que estão acima do limiar dado na variável threshold, supostamente sendo fontes de luz.
    '''
    #print("[INFO] limiarizando pela luminance com threshold = {}...".format(threshold))
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    Lchannel = img_HLS[:,:,LUMINANCE]
    mask = cv2.inRange(Lchannel, threshold, 1.0)
    res = cv2.bitwise_and(img, img, mask= mask)
    return res

def gaussian_bloom_mask(img):
    '''
    Borra uma imagem várias vezes utilizando filtro gaussiano com valores de sigma crescentes e então 
    soma esses resultados para criar uma máscara para aplicar o efeito de bloom.
    Parâmetros:
        img: numpy array da imagem original em três canais HLS, de dimensões largura x altura x 3.
    Retorna:
        Uma segunda imagem de três canais HLS com uma máscara para aplicar o efeito de bloom.
    '''
    #print("[INFO] criando bloom mask com filtro gaussiano...")
    new_img = img.copy()*0
    # A: com o filtro gaussiano, cada sigma é aplicado sobre uma nova cópia da imagem de entrada
    for blur in range(1, 5):
        temp_img = img.copy()
        temp_img = cv2.GaussianBlur(img, ksize = (0, 0), sigmaX = pow(2, blur))
        new_img += temp_img
        #cv2.imshow ('gaussian blur - sigma {}'.format(pow(2, blur)), temp_img)
        #cv2.waitKey (0)
    #cv2.imshow ('resultado gaussian bloom', new_img)
    #cv2.waitKey (0)
    return img
    #return new_img

def bloom(img, bloom_mask, weight):
    '''
    Aplica o efeito de bloom na imagem de entrada.
    Parâmetros:
        img: numpy array da imagem original em três canais HLS, de dimensões largura x altura x 3.
        bloom_mask: numpy array de uma máscara de bloom em três canais HLS, de dimensões largura x altura x 3.
        weight: peso atribuído entre máscara e imagem original. Quanto maior, mais forte o efeito de bloom.
    Retorna:
        Uma segunda imagem de três canais HLS com o efeito de bloom aplicado.
    '''
    # A: se não for aplicado algum tratamento de soma ponderada como esse, a imagem resultante da soma
    # com a máscara será ofuscante demais, estourando o limite do que é branco em grande parte da imagem.
    #print("[INFO] aplicando o efeito de bloom com peso {} para a mascara e peso {} para a imagem original...".format(weight, 1-weight))
    return img*(1-weight) + bloom_mask*weight

def adjust_gamma(img, gamma=1.0):
    """
    Aplica o ajuste gama através de uma lookup table que mapeia valores de pixels
    entre [0, 255] para seus valores gama ajustados.
    Entrada: img  : uma imagem dada por uma np.array do tipo uint8 BGR.
             gamma: valor gamma utilizado para a correção, float.
    Retorna: uma imagem dada por uma np.array do tipo uint8 BGR com valores alterados.
    """

    inv_gamma = 1.0 / gamma
    # A: criação da lookup table com o fator gama fornecido na entrada.
    table = np.array([((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # A: aplica a correção gama usando a lookup table com a função cv2.LUT.
    return cv2.LUT(img, table)

def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

def main():
    threshold = args["threshold"]
    weight = args["weight"]
    filename = args["input"].split(".")[0]
    # Abre a imagem;
    img = cv2.imread (args["input"])
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    rows, cols, channels = img.shape    

    if(not (0 < threshold < 1)):
        print ('Limiar deve estar no intervalo 0 < threshold <  1.\n')
        sys.exit ()

    if(not (0 < weight <= 0.5)):
        print ('Peso deve estar no intervalo 0 < threshold <=  0.5.\n')
        sys.exit ()

    # A: a bit depth pode ser adquirida pegando o dtype da np array e usando itemsize para
    # ver quantos bytes esse tipo tem.
    bpp = pow(256, img.dtype.itemsize) - 1

    img_lighter = adjust_gamma(img, 0.7)
    # A: convertendo a imagem para float32
    img = img.astype (np.float32) / bpp
    img_lighter = img_lighter.astype (np.float32) / bpp

    # cv2.imshow ('00 - imagem original', img)
    # cv2.waitKey (0)

    # A: aplicando a limiarização por brilho em HLS
    img_threshold = luminance_threshold(img_lighter, threshold)

    mask = 1 - img_threshold
    grey_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    th, grey_mask = cv2.threshold((grey_mask*bpp).astype(np.uint8), 128, 192, cv2.THRESH_OTSU)
    #cv2.imwrite(f"{filename} grey mask.png", grey_mask*bpp)
    r,c = np.where(grey_mask == 0)
    # mask = adjust_gamma((mask*bpp).astype(np.uint8), 0.5)
    # mask = mask.astype (np.float32) / bpp
    # img_lighter[(r,c)] = mask[(r, c)]

    bloom_mask = gaussian_bloom_mask(img_threshold)
    gaussian_bloom = bloom(img_lighter, bloom_mask, weight)
    mask = adjust_gamma((mask*bpp).astype(np.uint8), 0.1)
    mask = mask.astype (np.float32) / bpp
    gaussian_bloom[(r,c)] = mask[(r, c)]
    noise = mask*0
    noise = sp_noise(noise, 0.9) / 255
    output = gaussian_bloom + noise*0.1

    shape = img.shape
    width = shape[1]
    height = shape[0]
    
    output = cv2.resize(output, (int(width/3), int(height/3)))
    output = cv2.resize(output, (width, height))

    noise = sp_noise(noise, 0.9) / bpp
    output = gaussian_bloom + noise*0.1

    #bloom_mask = gaussian_bloom_mask(mask)
    #output = bloom(output, bloom_mask, 0.15)
    #output = adjust_gamma((output*bpp).astype(np.uint8), 0.5)

    cv2.imwrite(f"{filename}.jpg", output*bpp)

    #cv2.imshow ('01 - imagem limiarizada pela luminance com t = {}'.format(threshold), img_threshold)
    #cv2.waitKey (0)


if __name__ == "__main__":
    main()