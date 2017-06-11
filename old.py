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
num_epochs = 10
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
#    num_workers=2,

)
test_loader= Data.DataLoader(
    dataset= test_minist_dataset,
    batch_size = batch_size,
    shuffle = True,
 #   num_workers=2,

)

class LogisticRegression(nn.Module):
    def __init__(self, input_size,hidden_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size,num_classes)
    def forward(self, x):
        out1 = F.sigmoid(self.input(x))
        out2 = self.hidden(out1)
      # 	out = self.linear(x)
        return out2

model = LogisticRegression(input_size,hidden_size, num_classes)

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
#criterion = nn.MSELoss()  

criterion = nn.MSELoss()  
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)  
saveloss = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = torch.Tensor.float(images)
        images = Variable(images.view(-1,28*28))

        labels_onehot = torch.zeros(labels.size(0),num_classes)
        labels_onehot.scatter_(1,labels,1)

        labels_onehot = Variable(torch.Tensor.float(labels_onehot))

     #   print("epoch=",epoch,"i=",i,"images=",images.size(),"labels=",labels.size())
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
       # pdb.set_trace
       # print(outputs.size(),type(labels))
        loss = criterion(outputs, labels)
        saveloss.append(loss.data[0])
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
    
print('Accuracy of the model on the 10000 test images: %f %%'  %(100 * correct / total))


from matplotlib import pyplot as plt
xaxis = [x for x in range(600*num_epochs)]
plt.plot(xaxis,saveloss)
plt.show()

 #Save the Model
torch.save(model, 'model.pkl')