import torch
import torch.nn.functional as F
import torch.nn as nn
targets = torch.tensor([0, 0, 0, 3, 3, 3, 3, 0], device='cuda:0')
output = torch.tensor([[-1.0003, -1.1586, -0.8417,  1.7524],
                       [ 0.1564, -1.5598, -0.4407,  0.9209],
                       [ 0.1471, -0.3196, -0.9350,  0.2924],
                       [-1.7408, -1.2449, -0.9773,  2.0357],
                       [-1.4069, -1.8529, -0.6961,  2.2067],
                       [ 0.0118, -0.3171, -0.9144,  0.3762],
                       [-1.6195, -0.9323, -1.1269,  1.8752],
                       [ 5.6714, -3.8116, -1.9539, -2.1452]], device='cuda:0')

def cross_entrophy_loss(target, predicted_Output):
        softmax_ouput = F.softmax(predicted_Output, dim=1)
        loss = -torch.log(softmax_ouput[torch.arange(predicted_Output.size(0)),target]).mean()
        return loss
loss = cross_entrophy_loss(targets, output)

criterion = nn.CrossEntropyLoss()
loss_original = criterion(output, targets)

print(f"My Cross-entropy loss: {loss}")
print(f"My Cross-entropy loss: {loss_original}")
