import pytorch_lightning as pl
from byol_pytorch import BYOL
import local_dataset as DS
from torchvision import transforms, models, datasets
import os
import torch
from pytorch_metric_learning import testers

NUM_GPUS = 8
NUM_WORKERS = 80
LR = 3e-4
EPOCHS = 1000
BATCH_SIZE = 128
EPOCHS     = 1000
LR         = 3e-4
IMAGE_SIZE = 256

def get_all_embeddings(dataset, model):
    # dataloader_num_workers has to be 0 to avoid pid error
    # This only happens when within multiprocessing
    tester = testers.BaseTester(dataloader_num_workers=0)
    return tester.get_all_embeddings(dataset, model)

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        projection = self.learner(images,return_embedding = True)
        assert len(projection) == 2
        return projection[0] # return projections [1] for resnet output
    
    def embed(self, images):
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
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cwd =  os.getcwd()

    train_dataset = DS.RaccoonDataset(root=cwd,img_folder="../croppedImages/train",transforms = val_transform)
    val_dataset = DS.RaccoonDataset(root=cwd,img_folder="../croppedImages/test", transforms = val_transform)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
    # test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,num_workers=1)

    resnet = models.resnet50(pretrained=True)

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    model.load_state_dict(torch.load('./improved-net.pt'))
    model.cuda()
    result_train_img, result_train_label = get_all_embeddings(train_dataset,model)
    result_test_img, result_test_label = get_all_embeddings(val_dataset,model)
    
    torch.save(result_train_img.cpu(),'./train_imgs.pt')
    torch.save(result_test_img.cpu(),'./test_imgs.pt')

    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2, random_state=0,n_iter=10000,n_iter_without_progress=500,perplexity=35)
    embedding = [model.forward(sample[0].unsqueeze(dim=0).cuda()).detach() for sample in val_dataset]
    tsne = tsne_model.fit_transform(torch.cat(embedding,dim=0).cpu().detach().numpy())
    torch.save(torch.tensor(tsne),'./tsne.pt')
