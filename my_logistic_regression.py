#logistic_regression.py
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from input_file import getData,getLabel
import pdb
# Hyper Parameters 
input_size = 784
hidden_size = 26
num_classes = 10
num_epochs = 50
batch_size = 100
learning_rate = 0.01

train_x = torch.from_numpy(getData("train"))
train_y = torch.from_numpy(getLabel("train"))
test_x = torch.from_numpy(getData("test"))
test_y = torch.from_numpy(getLabel("test"))

train_minist_dataset = Data.TensorDataset( data_tensor=train_x,target_tensor=train_y)
test_minist_dataset = Data.TensorDataset(data_tensor=test_x,target_tensor=test_y)
train_loader = Data.DataLoader(
    dataset= train_minist_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers=2,

)
test_loader= Data.DataLoader(
    dataset= test_minist_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers=2,

)
'''
# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
'''
class LogisticRegression(nn.Module):
    def __init__(self, input_size,hidden_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size,num_classes)
    def forward(self, x):
        out1 = F.sigmoid(self.input(x))
        out2 = F.sigmoid(self.hidden(out1))
      # 	out = self.linear(x)
        return out2

model = LogisticRegression(input_size,hidden_size, num_classes)

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = torch.Tensor.float(images)
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels.view(100))
       # print("epoch=",epoch,"i=",i,"images=",images.size(),"labels=",labels.size())
       
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
                   % (epoch+1, num_epochs, i+1, len(train_minist_dataset)//batch_size, loss.data[0]))


# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = torch.Tensor.float(images)
    images = Variable(images.view(-1, 28*28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(model.state_dict(), 'model.pkl')
