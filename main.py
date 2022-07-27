import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# load data
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)
for X, y in test_dataloader:
	print(f"Shape of X [N, C, H, W]: {X.shape}")
	print(f"Shape of y: {y.shape} {y.dtype}")
	break

# define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# define model
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(X.shape[2]*X.shape[3], 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10)
		)

	def forward(self, x):
		x = self.flatten(x)
		return self.linear_relu_stack(x)

model = NeuralNetwork().to(device)

# define optimizer and loss
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)


#main train and test function
def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		pred = model(X)
		loss = loss_fn(pred, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
	num_test_data = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= num_test_data
	print(f"Test Error: \n Accurancy: {(100*correct):>0.1f}% Avg Loss: {test_loss:>7f}")


#train and test
epochs = 20
for t in range(epochs):
	print(f"Epoch {t+1}\n---------------------------------")
	train(train_dataloader, model, loss_fn, optimizer)
	test(test_dataloader, model, loss_fn)
print("Done!")


