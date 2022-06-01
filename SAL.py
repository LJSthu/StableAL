#-*-coding:utf-8-*-
import torch
from torch.autograd import grad
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
torch.autograd.set_detect_anomaly(True)

def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()

    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"



class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.9, epsilon=1e-8):
        self.device = torch.device("cuda:6")
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.m_hat = None
        self.v_hat = None
        self.initialize = False

    def update(self, grad, iternum, theta):
        if not self.initialize:
            self.m = (1-self.beta1)*grad
            self.v = (1 - self.beta2) * grad ** 2
            self.initialize = True
        else:
            assert self.m.shape == grad.shape
            self.m = self.beta1 * self.m + (1-self.beta1)*grad
            self.v = self.beta2 * self.v + (1-self.beta2)*grad**2

        self.m_hat = self.m / (1-self.beta1**iternum)
        self.v_hat = self.v / (1-self.beta2**iternum)
        return theta + self.lr * self.m_hat / (self.epsilon + torch.sqrt(self.v_hat))


class LinearModel(nn.Module):
    def __init__(self, num=1, num_classes = 1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(num, num_classes, bias=False)
        self.weight_init()

    def forward(self, data):
        output = self.linear(data)
        return output

    def weight_init(self):
        torch.nn.init.xavier_uniform(self.linear.weight)



class StableAL():
    def __init__(self, environments):
        self.weights = None
        self.model = None
        self.weight_grad = None
        self.xa_grad = None
        self.theta_grad = None
        self.gamma = None
        self.adversarial_data = None
        self.loss_criterion = torch.nn.MSELoss()

        self.X = None
        self.y = None
        self.environments = []
        for xe, ye in environments:
            if self.X is None:
                self.X = xe
                self.y = ye
            else:
                self.X = np.concatenate((self.X, xe), axis=0)
                self.y = np.concatenate((self.y, ye), axis=0)

            xe = torch.Tensor(xe)
            ye = torch.Tensor(ye)
            self.environments.append([xe, ye])
        self.X = torch.Tensor(self.X)
        self.y = torch.Tensor(self.y)



        # init
        dim_x = self.environments[0][0].size(1)
        print("model dim %d" % dim_x)
        self.model = LinearModel(dim_x)
        self.weights = torch.zeros(dim_x).reshape(-1,1)+100.0



    def cost_function(self, x, x_adv):
        cost = torch.mean(((x - x_adv) ** 2).mm(self.weights))
        return cost


    def r(self, environments, alpha=10.0):
        env_loss = []
        for x_e, y_e in environments:
            env_loss.append(self.loss_criterion(self.model(x_e), y_e))
        env_loss = torch.Tensor(env_loss)
        max_index = torch.argmax(env_loss)
        min_index = torch.argmin(env_loss)

        result = 0.0
        for idx, (x_e, y_e) in enumerate(environments):
            if idx == max_index:
                result += (alpha+1)*self.loss_criterion(self.model(x_e), y_e)
            elif idx == min_index:
                result += (1-alpha)*self.loss_criterion(self.model(x_e), y_e)
            else:
                result += self.loss_criterion(self.model(x_e),y_e)
        return result


    # generate adversarial data
    def attack(self, gamma, data, step):
        attack_lr = 7e-3
        images, labels = data
        images_adv = images.clone().detach()
        images_adv.requires_grad_(True)
        optimizer = Adam(learning_rate=attack_lr)

        for i in range(step):
            if images_adv.grad is not None:
                images_adv.grad.data.zero_()
            outputs = self.model(images_adv)
            loss = self.loss_criterion(
                outputs, labels) - gamma * self.cost_function(images, images_adv)
            loss.backward()
            images_adv.data = optimizer.update(images_adv.grad, i + 1, images_adv)


        self.weight_grad = -2*gamma*attack_lr*(images_adv - images)
        temp_image = images_adv.clone().detach()
        temp_label = labels.clone().detach()
        self.adversarial_data = (temp_image, temp_label)
        return images_adv, labels



    def train_theta(self, data, epochs, epoch_attack, gamma, end_flag = False):
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        for i in range(epochs):
            if i % 10 == 0 or not end_flag:
                images_adv, labels = self.attack(gamma, data, step=epoch_attack)
            else:
                images_adv, labels = self.attack(gamma, self.adversarial_data, step=epoch_attack)
            optimizer.zero_grad()
            outputs = self.model(images_adv)
            loss = self.loss_criterion(outputs, labels)
            

            if self.xa_grad is None:
                dloss_dtheta = grad(loss, self.model.parameters(), create_graph=True)[0].reshape(-1)
                dtheta_dx = []

                for j in range(dloss_dtheta.shape[0]):
                    dtheta_dx.append(grad(dloss_dtheta[j], images_adv, create_graph=True)[0])
                self.xa_grad = torch.stack(dtheta_dx,1)

            else:
                dloss_dtheta = grad(loss, self.model.parameters(), create_graph=True)[0].reshape(-1)
                dtheta_dx = []

                for j in range(dloss_dtheta.shape[0]):
                    dtheta_dx.append(grad(dloss_dtheta[j], images_adv, create_graph=True)[0])
                self.xa_grad += torch.stack(dtheta_dx, 1)

            if i % 1000 == 999:
                print('%d | %.4f | %s'%(i, loss, pretty(self.model.linear.weight)))


            loss.backward(retain_graph=True)
            optimizer.step()

        self.xa_grad *= (-0.01)


    def trainAll(self, epoch, epoch_theta, epoch_attack, epoch_w=1):
        min_weight = torch.min(self.weights)
        attack_gamma = (1.0 / min_weight).data
        deltaall = 20
        alpha = 0.5

        zero_list = []

        dim_x = self.environments[0][0].size(1)
        end_flag = False

        for t in range(epoch):
            self.model = LinearModel(dim_x)

            if t == epoch-1:

                epoch_theta = 5000
                epoch_attack = 100
                end_flag = True
                min_weight = torch.min(self.weights)
                attack_gamma = 10.0

            self.train_theta((self.X, self.y), epoch_theta, epoch_attack, attack_gamma, end_flag)

            rtheta = self.r(self.environments, alpha=alpha/math.sqrt(t+1))
            print("t = %d r %.4f"%(t, rtheta.data))

            self.theta_grad = grad(rtheta, self.model.parameters(), create_graph=True,allow_unused=True)[0]
            
            dr_dx = torch.matmul(self.theta_grad, self.xa_grad).squeeze()    
            deltaw = dr_dx*self.weight_grad
            deltaw = torch.sum(deltaw,0)


            deltaw[zero_list] = 0.0
            max_grad = torch.max(torch.abs(deltaw))
            deltastep = deltaall
            lr_weight = (deltastep / max_grad).detach()

            print('delta', deltaw)
            print('t = %d gamma=%.4f\n         weight %s\n         model %s' % (t, attack_gamma, pretty(self.weights.numpy()),
                                                            pretty(self.model.linear.weight)))

            # update w
            self.weights -= lr_weight * deltaw.detach().reshape(self.weights.shape)
            

            # adjust gamma according to min(weight)
            min_weight = 1e8
            for i in range(self.weights.shape[0]):
                if self.weights[i] > 0.0 and self.weights[i] < min_weight:
                    min_weight = self.weights[i]
                if self.weights[i] < 0.0:
                    self.weights[i] = 0.0
                    zero_list.append(i)


            attack_gamma = (1.0 / min_weight).data


    