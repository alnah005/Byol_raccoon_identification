import torch
from byol_pytorch import BYOL
import local_dataset as DS
import multiprocessing
import os
from torchvision import transforms, models, datasets
import pytorch_lightning as pl


NUM_GPUS = 8
NUM_WORKERS = 80
LR = 3e-4
EPOCHS = 1000
BATCH_SIZE = 64
EPOCHS     = 1000
LR         = 3e-4
IMAGE_SIZE = 128


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

if __name__ == "__main__":
    # Set the image transforms
    train_transform = transforms.Compose([
                                        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                        transforms.Lambda(lambda image: image.convert('RGB')),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

    val_transform = transforms.Compose([
                                        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                        transforms.Lambda(lambda image: image.convert('RGB')),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

    cwd =  os.getcwd()

    train_dataset = DS.RaccoonDataset(root=cwd,img_folder="../croppedImages/train",transforms = train_transform,byol=True)
    val_dataset = DS.RaccoonDataset(root=cwd,img_folder="../croppedImages/test", transforms = val_transform, byol = True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,num_workers=1)

    resnet = models.resnet50(pretrained=True)


    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = True,
        accelerator='ddp',
    )

    trainer.fit(model, train_loader)

    # save your improved network
    torch.save(model.state_dict(), './improved-net.pt')
