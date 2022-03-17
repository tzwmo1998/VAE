import os
from opt import config_parser
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import imageio
import models
import dataset
# import pdb
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

print("Imported all libraries successfully!")


cuda =  torch.cuda.is_available()


# TODO: Data parsing 

class AverageMeter(object):

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0 
        self.avg = 0 
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count


def unpreprocess(batch, shape=(1,3,1,1)):
    # to unnormalize image for visualization
    # data N C H W
    device = batch.device
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)
    return (batch - mean) / std

def save_ckpt(model, optimizer, epoch, args):
    ckpt_path = args.ckpt_dir
    os.makedirs(ckpt_path, exist_ok=True)
    path = f'{ckpt_path}/{epoch:02d}.tar'
    torch.save({
        "epoch": epoch, 
        "encoder_state_dict": model.encoder.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
        "optim_state_dict":  optimizer.state_dict(),
    }, path)
    
    print('Saved checkpoints at', path)

def gaussian_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale).cuda()
    mean = x_hat.cuda()
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))

def kl_divergence( z, p, q):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------

    #  get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl




def train(model, optimizer, train_loader, epoch, feat_size, args):
    model.train()
    loss_meter = AverageMeter() 

    train_loss = 0
    # get feature size from args
    B, C, H, W = feat_size
    for idx, batch in enumerate(train_loader):
        imgs = Variable(batch)
        if cuda:
            imgs = imgs.cuda()

        optimizer.zero_grad()

        z, xhat, p, q = model(imgs)
        
        recon_loss = nn.GaussianNLLLoss(x_hat, imgs)
        kl = kl_divergence(z, p, q)
        
        elbo = (kl - recon_loss)
        loss = elbo.mean()

        loss.backward()
        loss_meter.update(loss.item(), args.batch_size)
        optimizer.step()
        if idx % args.log_interval == 0:
            print('Train epoch: {} [{}/{}({:.0f}%)]\t Loss: {:.6f}'.format(
                  epoch, idx, len(train_loader),
                  100*idx/len(train_loader),
                  loss_meter.val))
            xhat = xhat.detach()
            xhat = torch.clamp(xhat, 0, 1)
            xhat = unpreprocess(xhat).permute(0,2,3,1)

            xhat = xhat[0].cpu()
           
            imgs = unpreprocess(imgs).permute(0,2,3,1)
            imgs = imgs[0].cpu()
            

    
            output_path = args.exp_dir+'/'+str(epoch)+'/'+str(idx)
            os.makedirs(output_path,exist_ok=True)
            path_test = f'{output_path}/{epoch:02d}_{idx:02d}.png'
            path_gt = f'{output_path}/{epoch:02d}_{idx:02d}_{"gt"}.png'
            img = xhat.numpy()
            img_gt = imgs.numpy()
            imageio.imwrite(path_test, (img*255).astype('uint8'))
            imageio.imwrite(path_gt, (img_gt*255).astype('uint8'))

        if (idx + 1) % 100 == 0:
            save_ckpt(model, optimizer, epoch, args)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
         epoch, loss_meter.avg))
    
    



def validation(model, optimizer, test_loader, epoch, args):
    loss_meter = AverageMeter() 
    model.eval()
    for idx, batch in enumerate(test_loader):
        imgs = Variable(batch)
        if cuda:
            imgs = imgs.cuda()
        with torch.no_grad():
            z, xhat, p, q = model(imgs)
        
            recon_loss = gaussian_likelihood(xhat, model.log_scale, x)
            kl = kl_divergence(z, p, q)
        
            elbo = (kl - recon_loss)
            loss = elbo.mean()
            
        
        loss_meter.update(loss.item(), args.batch_size)
        print('Test epoch: {} Loss: {:.6f}'.format(epoch, loss_meter.val))



if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    train_dataset, val_dataset = dataset.get_data_loaders(args)
    train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=args.batch_size,
                          pin_memory=True)
    test_loader = DataLoader(val_dataset,
                          shuffle=False,
                          batch_size=args.batch_size,
                          pin_memory=True)

    feat_size = (args.batch_size, 3, args.img_downscale*512, args.img_downscale*640)
    
    if args.load_ckpt:
        checkpoint = torch.load(args.ckpt_dir)
        autoencoder = models.VAE(args, feat_size, checkpoint)
       
    else:
        autoencoder = models.VAE(args, feat_size)
    optimizer = optim.Adam(autoencoder.parameters(), lr = args.lrate)
    if args.load_ckpt:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if cuda:
        autoencoder.cuda()

    for epoch in range(args.epochs):
        print("start_epoch")
        train(autoencoder, optimizer, train_loader, epoch, feat_size, args)
        validation(autoencoder, optimizer, test_loader, epoch, args)

    save_ckpt(autoencoder, optimizer, args.epochs, args)