from models import load_model
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

EVAL_DATA = "predictions/"
def predict():
    f = load_model()
    preds = []
    for i in range(28051):
        if i % 1000 == 0:
            print(i)    
        im_name = '%0*d' % (5, i+1) + ".jpg"
        f.eval()
        x = Image.open(EVAL_DATA+im_name)
        transform=transforms.ToTensor()
        x = transform(x)
        x = x.view(1,1,20,20)
        
        y = f(x)
        # preds.append(int(torch.max(y.data, 1)[1].numpy()))
        preds.append(F.softmax(y,dim=1))
        print(F.softmax(y,dim=1))
    
    # predictions = pd.DataFrame(data = preds, columns = ["Y"], index = range(0,28051))
    # predictions.to_csv("predictions.csv")
if __name__ == '__main__':
    predict()