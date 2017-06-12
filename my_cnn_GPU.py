import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.utils.data as Data
from input_file import getData,getLabel
import pdb
import time

t = time.time()
INPUT_SIZE = 28*28
CLASS_NUM = 10
BATCH_SIZE = 100
EPOCH = 20

train_x = torch.from_numpy(getData("train")/255.)
train_y = torch.from_numpy(getLabel("train"))
test_x = torch.from_numpy(getData("test")/255.)
test_y = torch.from_numpy(getLabel("test"))

train_minist_dataset = Data.TensorDataset( data_tensor=train_x,target_tensor=train_y)
test_minist_dataset = Data.TensorDataset(data_tensor=test_x,target_tensor=test_y)


train_loader = Data.DataLoader(
    dataset= train_minist_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,

)
test_loader= Data.DataLoader(
    dataset= test_minist_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
)

class CNN(nn.Module):     
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(1,16,5,1,2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
        nn.Conv2d(16,32,5,1,2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )

        self.out = nn.Linear(32*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) 
        out = self.out(x)
        return out


cnn = CNN()
cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(),)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for setp,(x,y) in enumerate(train_loader):
        x = torch.Tensor.float(x)
        b_x = Variable(x.view(BATCH_SIZE,1,28,28)).cuda()
        b_y = Variable(y.view(100)).cuda()
       # print(b_x,b_y)
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (setp+1) % 100 == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
                   % (epoch+1, EPOCH, setp+1, len(train_minist_dataset)//BATCH_SIZE, loss.data[0]))


correct = 0
total = 0
for images, labels in test_loader:
    images = torch.Tensor.float(images)
    images = Variable(images.view(BATCH_SIZE,1,28,28)).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    
print('Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))

print("time cost:",time.time()-t)
