import pytorch_lightning as pl
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F 
import torch
import models
import dataset
from opt import config_parser
import os
import imageio



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAESystem(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.feat_size = (args.batch_size, 3,  args.img_downscale*512, args.img_downscale*640)
        # encoder, decoder
        if args.load_ckpt:
            self.checkpoint = torch.load(args.ckpt_dir+'/32.tar')
        else:
            self.checkpoint = None
        self.args = args
        if self.checkpoint is not None:
            model = models.VAE(args, self.feat_size, checkpoint=self.checkpoint)
        else:
            model = models.VAE(args, self.feat_size)
        self.model = model

    def prepare_data(self):
        print("data loaded")
        self.train_dataset, self.val_dataset = dataset.get_data_loaders(self.args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.args.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          batch_size=self.args.batch_size,
                          pin_memory=True)

    def unpreprocess(self, data, shape=(1,3,1,1)):
        # to unnormalize image for visualization
        # data N C H W
#         shape = (self.args.batch_size, 3, 1, 1)
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)
        return (data-mean) / std
    
    def img2mse(self, x, y):
        return torch.mean((x - y) ** 2)
    
    def mse2psnr(self, mse):
        return -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(device))

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lrate)
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optim_state_dict'])
        return self.optimizer


    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(x_hat, scale)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))


    def kl_divergence(self, z, p, q):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
#         return kl.mean()


    def training_step(self, batch, batch_idx):
        loss = 0
        imgs = batch

        # encode x to get the mu and variance parameters
        z, x_hat, p, q = self.model(imgs)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.model.log_scale, imgs)
#         recon_loss = F.mse_loss(x_hat, imgs)
        # kl
        kl = self.kl_divergence(z, p, q)
#         kl = self.kl_divergence(z, p, q)*88*343*self.args.kl_coeff
        # elbo
        elbo = (kl - recon_loss)
        loss = elbo.mean()
        

#         

#         kl = self.kl_divergence(z, p, q)
#         kl = kl*343*88/self.args.batch_size

#         elbo = self.args.kl_coeff*kl + recon_loss
#         loss = elbo.mean()
        reconstruction = x_hat.detach()

        reconstruction = self.unpreprocess(reconstruction).permute(0,2,3,1)
        reconstruction = torch.clamp(reconstruction, 0, 1)
        imgs = self.unpreprocess(imgs).permute(0,2,3,1)
        mse = self.img2mse(imgs, reconstruction)
        psnr = self.mse2psnr(mse)
        if self.current_epoch%1==0 and batch_idx==100:
            imgs = imgs.cpu()
            reconstruction = reconstruction.cpu()
            output_path = args.exp_dir+'/'+str(self.current_epoch+53)
            os.makedirs(output_path,exist_ok=True)
            for i in range(batch.size(0)//2):
                path_test = f'{output_path}/{self.current_epoch+53:02d}_{i:02d}.png'
                path_gt = f'{output_path}/{self.current_epoch+53:02d}_{i:02d}_{"gt"}.png'
                img = reconstruction[i].numpy()
                img_gt = imgs[i].numpy()
                imageio.imwrite(path_test, (img*255).astype('uint8'))
                imageio.imwrite(path_gt, (img_gt*255).astype('uint8'))
                
            self.save_ckpt()

        self.log_dict({
            'loss': loss,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'mse': mse.mean(),
            'psnr': psnr.mean()
        })
        


        return {"loss":loss}

    def validation_step(self, batch, batch_idx):
        # encode x to get the mu and variance parameters
        imgs = batch
        z, x_hat, p, q = self.model(imgs)

# #         # reconstruction loss
#         recon_loss = self.gaussian_likelihood(x_hat, self.model.log_scale, imgs)

#         # kl
#         kl = self.kl_divergence(z, p, q)

#         # elbo
#         elbo = (kl - recon_loss)
#         loss = elbo.mean()
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.model.log_scale, imgs)
#         recon_loss = F.mse_loss(x_hat, imgs)
        # kl
        kl = self.kl_divergence(z, p, q)
#         kl = self.kl_divergence(z, p, q)*88*343*self.args.kl_coeff
        # elbo
        elbo = (kl - recon_loss)
        loss = elbo.mean()



        
        x_hat = self.unpreprocess(x_hat).permute(0,2,3,1)
        x_hat = torch.clamp(x_hat, 0, 1)
        imgs = self.unpreprocess(imgs).permute(0,2,3,1)
        mse = self.img2mse(imgs, x_hat)
        psnr = self.mse2psnr(mse)
        
        imgs = imgs.cpu()
        x_hat = x_hat.cpu()
        output_path = args.exp_dir+'/'+str(self.current_epoch+53)+'/val'
        os.makedirs(output_path,exist_ok=True)
        if batch_idx==0:
            for i in range(batch.size(0)):
                path_test = f'{output_path}/{self.current_epoch+53:02d}_{i:02d}.png'
                path_gt = f'{output_path}/{self.current_epoch+53:02d}_{i:02d}_{"gt"}.png'
                img = x_hat[i].numpy()
                img_gt = imgs[i].numpy()
                imageio.imwrite(path_test, (img*255).astype('uint8'))
                imageio.imwrite(path_gt, (img_gt*255).astype('uint8'))
            
        self.log_dict({
            'val/loss': loss,
            'val/kl': kl.mean(),
            'val/recon_loss': recon_loss.mean(),
            'val/mse': mse.mean(),
            'val/psnr': psnr.mean()
        })


    def save_ckpt(self):
        ckpt_path = self.args.ckpt_dir
        os.makedirs(ckpt_path, exist_ok=True)
        path = f'{ckpt_path}/{self.current_epoch+53:02d}.tar'
        torch.save({
            "epoch": self.current_epoch, 
            "encoder_state_dict": self.model.encoder.state_dict(),
            "decoder_state_dict": self.model.decoder.state_dict(),
            "log_scale":self.model.log_scale,
            "optim_state_dict":  self.optimizer.state_dict(),
        }, path)

        print('Saved checkpoints at', path)



if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    system = VAESystem(args)
    
#     checkpoint_callback = ModelCheckpoint(os.path.join(f'log/{args.expname}/ckpts/','{epoch:02d}'),
#                                           monitor='val/PSNR',
#                                           mode='max',
#                                           save_top_k=2)

    os.makedirs(args.log_dir, exist_ok=True)
    logger = loggers.TestTubeLogger(
        save_dir=args.log_dir,
        name=args.expname,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(gpus=args.num_gpus, max_epochs=args.epochs,
                      logger=logger)

    trainer.fit(system)
    system.save_ckpt()
    torch.cuda.empty_cache()