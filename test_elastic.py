from elasticity_op import elastic, elasticity, OptimizeElasticity, relevance_elastic
import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.e1 = elasticity(10, relevance=True)
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        a = torch.nn.functional.relu(x)
        print(type(a))
        print(type(x))
        return self.e1(a)

def test_elastic():
    input = torch.autograd.Variable(torch.Tensor(2, 10), requires_grad=True)
    psi = torch.autograd.Variable(torch.ones(10)*3, requires_grad=True)
    y = relevance_elastic.apply(input, psi)    
    y.backward(torch.Tensor(2, 10))

def test_elasticity():
    input = torch.autograd.Variable(torch.ones(128, 10)*2, requires_grad=True)
    incomming_grad = torch.Tensor(128, 10)
    model = Net()
    out = model(input)

    out.backward(torch.ones(128, 10)*4)
    print(input.grad)
    # print(out)
    # print(model.e1.psi.grad)

def test_optimization():
    input = torch.autograd.Variable(torch.ones(2, 10)*2, requires_grad=True)
    incomming_grad = torch.Tensor(2, 10)
    model = Net()
    out = model(input)
    opt = OptimizeElasticity(model.parameters(), lr=0.05)

    print(next(model.parameters()))
    out.backward(torch.ones(2, 10)*4)
    opt.step(is_abs=True)
    print(next(model.parameters()))
    

if __name__ == "__main__":
    # test_elastic()
    # test_elasticity()
    test_optimization()