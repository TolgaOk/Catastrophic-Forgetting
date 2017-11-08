from ..elasticity_ops import elastic, elasticity, OptimizeElasticity, relevance_elastic, elastic_linear
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

class Net2(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net2, self).__init__()
        self.fc = elastic_linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

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
    
def test_weigt_only_elastic():

    input = torch.autograd.Variable(torch.ones(2, 4), requires_grad=True)
    incomming_grad = torch.ones(2, 3)*-3.0
    model = Net2(4, 3)

    out = model(input)
    out.backward(incomming_grad)

    print(model.fc.elastic_weight.psi.grad)
    print(model.fc.elastic_weight.psi.data)
    print(model.fc.weight.data)

if __name__ == "__main__":
    # test_elastic()
    # test_elasticity()
    # test_optimization()
    test_weigt_only_elastic()