import numpy as np

# Tools for generation structural noise given an image size.

def sinc(input,f,decay):
    return np.sin(f*input)/(1+input*decay)

def circleConcentric(X,Y):
    r_min = 3/4*np.sqrt(X**2 + Y**2)
    r_max = 4/3*r_min
    r = np.random.uniform(r_min, r_max)
    theta = np.random.uniform(0, 2 * np.pi)
    return [X/2+r*np.cos(theta), Y/2+r*np.sin(theta)]



def struc_image(IMAGE_SIZE_X, IMAGE_SIZE_Y, amplitude):
    sources = np.random.randint(2, 4)
    diag = np.sqrt(IMAGE_SIZE_X**2 + IMAGE_SIZE_Y**2)

    ImageSum = []
    for _ in range(sources):
        frequency = np.random.uniform(10*np.pi/diag, 12*np.pi/diag)
        decay = np.random.uniform(1,5)*1e-3
        pos = circleConcentric(IMAGE_SIZE_X, IMAGE_SIZE_Y)
        r = (int(pos[0]), int(pos[1]))
        x = np.linspace(0, IMAGE_SIZE_X, IMAGE_SIZE_X)
        y = np.linspace(0, IMAGE_SIZE_Y, IMAGE_SIZE_Y)
        xv, yv = np.meshgrid(x, y)
        R = np.sqrt((xv - r[0]) ** 2 + (yv - r[1]) ** 2)
        Image = amplitude * sinc(R, frequency, decay) * np.exp(1j*frequency*R)
        ImageSum.append(Image)

    for i in range(sources):
        if i == 0:
            imageSummed = ImageSum[i]
        else:
            imageSummed += ImageSum[i]

    return imageSummed
