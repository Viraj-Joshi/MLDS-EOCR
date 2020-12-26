from models import load_model
from utils import generate_matrices
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
        preds.append(int(torch.max(y.data, 1)[1].numpy()))
        # preds.append(F.softmax(y,dim=1))
        # print(F.softmax(y,dim=1))
    
    predictions = pd.DataFrame(data = preds, columns = ["Y"], index = range(0,28051))
    predictions.to_csv("predictions_orig.csv")

    # Apply Viterbi algorithm (log variant)
    A,C = generate_matrices()
    # B = [
    #     [1] + [0] * 42,
    #     [0] + [1] + [0]*42,
    #     [0]*2 + [1] + [0]*40,
    #     [0]*3 + [.93] +


    # ]
    B = A
    O = preds
    S_opt, D_log, E = viterbi_log(A, C, B, O)

    predictions = pd.DataFrame(data = S_opt, columns = ["Y"], index = range(0,28051))
    predictions.to_csv("predictions_HMM.csv")

def viterbi_log(A, C, B, O):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A: State transition probability matrix of dimension I x I
        C: Initial state distribution  of dimension I
        B: Output probability matrix of dimension I x K
        O: Observation sequence of length N

    Returns:
        S_opt: Optimal state sequence of length N
        D_log: Accumulated log probability matrix
        E: Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_log = np.log(B + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_log[i, O[n]]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, 0, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt, D_log, E


if __name__ == '__main__':
    predict()