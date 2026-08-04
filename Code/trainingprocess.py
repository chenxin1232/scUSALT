import torch
import torch.optim as optim
from torch.autograd import Variable
import os
from scipy.linalg import norm
from Code.data_load import PrepareDataloader
from Code.model import Embedding_layer,Prediction_layer
from Code.loss import L1regularization, CellLoss, EncodingLoss
import torch.nn.functional as F
from Code.utils import *
from setting import Setting
setting = Setting()

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grl(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DomainDiscriminator(torch.nn.Module):
    """模态判别器：区分 embedding 来自 RNA(0) 还是 ATAC(1)"""
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2)
        )

    def forward(self, z):
        return self.net(z)

def def_cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
            
def prepare_input(data_list, setting):
    output = []
    for data in data_list:
        output.append(Variable(data.to(setting.device)))
    return output

class Training():
    def __init__(self, setting):
        self.setting = setting
        # load data
        self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders, self.test_atac_loaders, self.training_iters = PrepareDataloader(setting).getloader()
        self.training_iteration = 0
        for atac_loader in self.train_atac_loaders:
            self.training_iteration += len(atac_loader)
        
        # initialize dataset       
        if self.setting.use_cuda:  
            self.model_encoder = torch.nn.DataParallel(Embedding_layer(setting.input_size).to(self.setting.device)) #embedding layer
            self.model_cell = torch.nn.DataParallel(Prediction_layer(setting.number_of_class).to(self.setting.device)) #prediction layer
        else:
            self.model_encoder = Embedding_layer(setting.input_size).to(self.setting.device)
            self.model_cell = Prediction_layer(setting.number_of_class).to(self.setting.device)
                
        # initialize criterion (loss)
        self.criterion_cell = CellLoss()
        self.criterion_encoding = EncodingLoss(dim=setting.embedding_size, use_gpu = self.setting.use_cuda)
        self.l1_regular = L1regularization()

        # 对抗对齐（分布对齐）：始终启用
        self.lambda_adv = getattr(self.setting, "lambda_adv", 0.1)
        self.adv_warmup_epochs = getattr(self.setting, "adv_warmup_epochs", 0)
        if self.setting.use_cuda:
            self.domain_disc = torch.nn.DataParallel(
                DomainDiscriminator(in_dim=self.setting.embedding_size).to(self.setting.device)
            )
        else:
            self.domain_disc = DomainDiscriminator(in_dim=self.setting.embedding_size).to(self.setting.device)
        self.optimizer_domain = optim.SGD(self.domain_disc.parameters(), lr=self.setting.lr, momentum=self.setting.momentum, weight_decay=0)
        
        # initialize optimizer (sgd/momemtum/weight decay)
        self.optimizer_encoder = optim.SGD(self.model_encoder.parameters(), lr=self.setting.lr, momentum=self.setting.momentum,
                                           weight_decay=0)
        self.optimizer_cell = optim.SGD(self.model_cell.parameters(), lr=self.setting.lr, momentum=self.setting.momentum,
                                        weight_decay=0)
        
    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.setting.lr * (0.1 ** ((epoch - 0) // self.setting.lr_decay_epoch))
        if (epoch - 0) % self.setting.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def load_checkpoint(self, args):
        if self.setting.checkpoint is not None:
            if os.path.isfile(self.setting.checkpoint):
                print("=> loading checkpoint '{}'".format(self.setting.checkpoint))
                checkpoint = torch.load(self.setting.checkpoint)                
                self.model_encoder.load_state_dict(checkpoint['model_encoding_state_dict'])
                self.model_cell.load_state_dict(checkpoint['model_cell_state_dict'])
            else:
                print("=> no resume checkpoint found at '{}'".format(self.setting.checkpoint))
                
    def train(self, epoch):
        self.model_encoder.train()
        self.model_cell.train()
        self.domain_disc.train()
        total_encoding_loss, total_rna_loss,total_atac_loss,total_cell_loss, total_sample_loss, total_kl_loss = 0., 0., 0., 0., 0., 0.
        self.adjust_learning_rate(self.optimizer_encoder, epoch)
        self.adjust_learning_rate(self.optimizer_cell, epoch)

        # initialize iterator
        iter_rna_loaders = []
        iter_atac_loaders = []
        for rna_loader in self.train_rna_loaders:
            iter_rna_loaders.append(def_cycle(rna_loader))
        for atac_loader in self.train_atac_loaders:
            iter_atac_loaders.append(def_cycle(atac_loader))
            
            
        # Enter batch    
        for batch_idx in range(self.training_iters):
            # rna forward
            rna_embeddings = []
            rna_cell_predictions = []
            rna_labels = []
            rna_embeddings_2 = []
            rna_embeddings_3= []
            for iter_rna_loader in iter_rna_loaders:
                
                # prepare data batch_size = 256
                rna_data, rna_label = next(iter_rna_loader)    
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.setting) 
                # model forward
                rna_embedding = self.model_encoder(rna_data) 
                rna_embeddings.append(rna_embedding)
                rna_labels.append(rna_label)            
            
                rna_cell_prediction = self.model_cell(rna_embedding) 
                rna_cell_predictions.append(rna_cell_prediction)
                
                

                
            # atac forward
            atac_embeddings = []
            atac_cell_predictions = []
            for iter_atac_loader in iter_atac_loaders:
                atac_data = next(iter_atac_loader)    
                # prepare data
                atac_data = prepare_input([atac_data], self.setting)[0] 
                
                atac_all = atac_data
                peak_data = None
                if len(setting.peak_paths) > 0:
                    split_index = rna_data.size()[2]
                    atac_data = atac_all[:, :, :split_index]
                    peak_data = atac_all[:, :, split_index:]  
                    peak_data = peak_data.view(peak_data.size()[0], peak_data.size()[2])
                
                
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                atac_embeddings.append(atac_embedding)                                            
                atac_cell_prediction = self.model_cell(atac_embedding)
                atac_cell_predictions.append(atac_cell_prediction)
        
            
            # loss
            encoding_loss = self.criterion_encoding(atac_embeddings, rna_embeddings,rna_labels,peak_data)
            regularization_loss_encoder = self.l1_regular(self.model_encoder)            

            # 对抗对齐损失：用当前 batch 的 RNA/ATAC embedding 做模态判别
            # warmup：前 adv_warmup_epochs 轮 lambd=0
            lambd = self.lambda_adv if epoch >= self.adv_warmup_epochs else 0.0
            z_rna = rna_embeddings[0]
            z_atac = atac_embeddings[0]
            z = torch.cat([z_rna, z_atac], dim=0)
            y_domain = torch.cat([
                torch.zeros(z_rna.size(0), dtype=torch.long, device=z.device),
                torch.ones(z_atac.size(0), dtype=torch.long, device=z.device),
            ], dim=0)
            logits = self.domain_disc(grl(z, lambd=lambd))
            adv_loss = F.cross_entropy(logits, y_domain)
            
            cell_loss = self.criterion_cell(rna_cell_predictions[0], rna_labels[0]) #torch.Size([256, 7]),torch.Size([256])
            for i in range(1, len(rna_cell_predictions)):
                cell_loss += self.criterion_cell(rna_cell_predictions[i], rna_labels[i])
            cell_loss = cell_loss/len(rna_cell_predictions)
            regularization_loss_cell = self.l1_regular(self.model_cell)
            
            # update encoding weights
            self.optimizer_encoder.zero_grad()
            self.optimizer_domain.zero_grad()
            regularization_loss_encoder.backward(retain_graph=True)
            total_encoding_loss_with_adv = encoding_loss + adv_loss
            total_encoding_loss_with_adv.backward(retain_graph=True)
                    
            # update cell weights
            self.optimizer_cell.zero_grad()
            cell_loss.backward(retain_graph=True)       
            regularization_loss_cell.backward(retain_graph=True)
            
            self.optimizer_encoder.step()
            self.optimizer_cell.step()
            self.optimizer_domain.step()

            # print log
            total_encoding_loss += float(total_encoding_loss_with_adv.data.item())
            total_cell_loss += cell_loss.data.item()

            
            progress_bar(batch_idx, self.training_iters,
                          'Embedding loss: %.3f , Prediction loss:  %.3f,'% (
                              total_encoding_loss / (batch_idx + 1), total_cell_loss / (batch_idx + 1),
                              ))
            
            
        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_cell_state_dict': self.model_cell.state_dict(),
            'model_encoding_state_dict': self.model_encoder.state_dict(),
            'optimizer': self.optimizer_cell.state_dict()            
        })
        
    def write_embeddings(self):
        self.model_encoder.eval()
        self.model_cell.eval()
        if not os.path.exists("output_scUSALT/"):
            os.makedirs("output_scUSALT/")
        
        # rna db
        for i, rna_loader in enumerate(self.test_rna_loaders):
            db_name = os.path.basename(self.setting.rna_paths[i]).split('.')[0]
            fp_em = open('./output_scUSALT/' + db_name + '_embeddings.txt', 'w')
            for batch_idx, (rna_data, rna_label) in enumerate(rna_loader):    
                # prepare data
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.setting)
                    
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                            
                rna_embedding = rna_embedding.data.cpu().numpy()
                
                # normalization & softmax
                rna_embedding = rna_embedding / norm(rna_embedding, axis=1, keepdims=True)
                                
                # write embeddings
                test_num, embedding_size = rna_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(rna_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(rna_embedding[print_i][print_j]))
                    fp_em.write('\n')                    
                
                progress_bar(batch_idx, len(rna_loader),
                         'write embeddings for db:' + db_name)                    
            fp_em.close()
        
        
        # atac db
        for i, atac_loader in enumerate(self.test_atac_loaders):
            db_name = os.path.basename(self.setting.atac_paths[i]).split('.')[0]
            fp_em = open('./output_scUSALT/' + db_name + '_embeddings.txt', 'w')
            # fp_pre = open('./output_scUSALT/' + db_name + '_predictions.txt', 'w')
            for batch_idx, (atac_data) in enumerate(atac_loader):    
                # prepare data
                atac_data = prepare_input([atac_data], self.setting)[0]
                # atac_all = atac_data
                # if len(setting.peak_paths) > 0:
                #     split_index = rna_data.size()[2]
                #     atac_data = atac_all[:, :, :split_index]
                
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                atac_cell_prediction = self.model_cell(atac_embedding)
                                
                                
                atac_embedding = atac_embedding.data.cpu().numpy()
                atac_cell_prediction = atac_cell_prediction.data.cpu().numpy()
                
                # normalization & softmax
                atac_embedding = atac_embedding / norm(atac_embedding, axis=1, keepdims=True)
                atac_cell_label = atac_cell_prediction.argmax(1)
                
                # write embeddings
                test_num, embedding_size = atac_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(atac_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(atac_embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
                # write predictions
                # test_num, prediction_size = atac_cell_label.reshape(-1, 1).shape
                # for print_i in range(test_num):
                #     fp_pre.write(str(atac_cell_label[print_i]))
                #     for print_j in range(1, prediction_size):
                #         fp_pre.write(' ' + str(atac_cell_label[print_i]))
                #     fp_pre.write('\n')
                
                progress_bar(batch_idx, len(atac_loader),
                         'write embeddings for db:' + db_name)                    
            fp_em.close()
            # fp_pre.close()       
