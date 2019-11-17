import torch
import argparse
from utils import load_data, plot_graph
from model import Model
import pandas as pd 
import numpy as np 

#设置参数
parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.1, help="learning rate")
parse.add_argument('--input_size', type=int, default=3, help="input dim")
parse.add_argument('--hidden-size', type=int, default=50, help="hidden dim")
parse.add_argument('--time_step', type=int, default=7, help="day length")
parse.add_argument('--predict_size', type=int, default=1, help="predict day lenght")
parse.add_argument('--epochs', type=int, default=2000, help="iter number")

args = parse.parse_args()

def train():
    x_data, y_data = load_data(days=args.time_step, input_size=args.input_size)
    model = Model(args.input_size, args.hidden_size, args.predict_size, args.time_step)
    optim = torch.optim.SGD(
      params=model.parameters(),
      lr=args.lr,
      momentum=0.9
    )#动量梯度下降器
    lossFuc = torch.nn.MSELoss()  #损失函数为均方根
    if  torch.cuda.is_available: ## 判断是否有gpu
        x_data = x_data.cuda()
        y_data =y_data.cuda()
        model = model.cuda()

    for epoch in range(args.epochs):
        optim.zero_grad() #梯度清零
        output = model.forward(x_data)
        cost = lossFuc(output, y_data)
        cost.backward() #反向传播
        optim.step()  #梯度更新
        print(
          "epoch: {}".format(epoch),
          "loss: {}".format(cost.item())
        )
    torch.save(model, r"model.pkl") #保存模型

def test():
    x_data, y_data = load_data(isTraining=False)
    model = torch.load(r"model.pkl")
    lossFuc = torch.nn.MSELoss()
    output = model.forward(x_data)
    loss = lossFuc(output, y_data)
    print("Test loss:{}".format(loss.item()))

def main():
    #train()
    test()

if __name__ == "__main__":
    main()