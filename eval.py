from train import load_model
from torchvision import transforms
from PIL import Image
import numpy as np

EVAL_DATA = "predictions/"
def predict():
    f = load_model()

    for i in range(28051):    
        im_name = '%0*d' % (5, i+1) + ".jpg"
        x = Image.open(EVAL_DATA+im_name)
        transform=transforms.ToTensor()

        x = transform(img)
        y = f(x)
        arr = y.data.cpu().numpy()
        # write CSV
        # np.savetxt('output.csv', arr)
        print(arr)