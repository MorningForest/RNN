import torch 


class Model(torch.nn.Module):
    '''定义rnn'''
    def __init__(self, in_feature, hidden_feature, output_class, time_step):
        super(Model, self).__init__()
        self.in_feature = in_feature    #输入向量维度
        self.hidden_feature = hidden_feature  #隐藏特征维度
        self.time_step = time_step  #时间步长
        self.rnn = torch.nn.RNN(
          input_size=in_feature,
          hidden_size=hidden_feature,
          batch_first=True 
        )
        self.fct = torch.nn.Linear(
          hidden_feature, output_class
        )#全连接层

    def forward(self, inputs):
        inputs = inputs.view(-1, self.time_step, self.in_feature)
        outputs, _ = self.rnn(inputs) # 输出为所以节点预测值，和隐藏值
        #选取最后一个时间节点的输出
        output = torch.nn.functional.tanh(
          self.fct(outputs[:,-1,:])
        )
        return output
