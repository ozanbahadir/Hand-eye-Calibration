import torch
import torch.nn as nn
labels = torch.tensor([1,1,1])
#labels = labels.argmax(1)
criterion = nn.MSELoss(reduction='sum')
x = torch.randn(3, 3, requires_grad=True)

loss = criterion(x, labels)
loss.backward()

print(labels)
print(x)
print(loss)