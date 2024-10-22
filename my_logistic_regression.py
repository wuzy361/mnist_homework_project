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
hidden1_size = 500
hidden2_size = 300
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001

train_x = torch.from_numpy(getData("train")/255.)
train_y = torch.from_numpy(getLabel("train"))
test_x = torch.from_numpy(getData("test")/255.)
test_y = torch.from_numpy(getLabel("test"))

train_minist_dataset = Data.TensorDataset( data_tensor=train_x,target_tensor=train_y)
test_minist_dataset = Data.TensorDataset(data_tensor=test_x,target_tensor=test_y)
train_loader = Data.DataLoader(
    dataset= train_minist_dataset,
    batch_size = batch_size,
    shuffle = True,
   # num_workers=2,

)
test_loader= Data.DataLoader(
    dataset= test_minist_dataset,
    batch_size = batch_size,
    shuffle = True,
    #num_workers=2,

)

class LogisticRegression(nn.Module):
    def __init__(self, input_size,hidden1_size,hidden2_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.input = nn.Linear(input_size, hidden1_size)
        self.hidden1 = nn.Linear(hidden1_size,hidden2_size)
        self.hidden2 = nn.Linear(hidden2_size,num_classes)
    def forward(self, x):
        out1 = F.relu(self.input(x))
        out2 = F.relu(self.hidden1(out1))
        out3 = self.hidden2(out2)
      #     out = self.linear(x)
        return out3

model = LogisticRegression(input_size,hidden1_size, hidden2_size, num_classes)

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)  

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
    
print('Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))

# Save the Model
torch.save(model.state_dict(), 'model.pkl')