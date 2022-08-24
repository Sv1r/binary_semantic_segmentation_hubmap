import torch
import random
import numpy as np
import segmentation_models_pytorch

import utils
import models
import dataset
import settings

# Fix random
random.seed(settings.RANDOM_STATE)
np.random.seed(settings.RANDOM_STATE)
torch.manual_seed(settings.RANDOM_STATE)
torch.cuda.manual_seed(settings.RANDOM_STATE)
# Model
INITIAL_MODEL = 'MaNet'
model = models.model_dict[INITIAL_MODEL]
# Freeze Encoder
for param in model.encoder.parameters():
    param.requires_grad = False
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE)
# Losses
loss_tversky = segmentation_models_pytorch.losses.TverskyLoss(mode='binary', log_loss=True, alpha=.3, beta=.7)
loss_focal = segmentation_models_pytorch.losses.FocalLoss(mode='binary')


def loss(preds, mask):
    """Combo loss function"""
    focal = loss_focal(preds, mask)
    tversky = loss_tversky(preds, mask)
    return torch.mean(focal + tversky)


# Train Model
model, loss_and_metrics = utils.train_model(
    model=model,
    train_dataloader=dataset.train_dataloader,
    valid_dataloader=dataset.valid_dataloader,
    loss=loss,
    optimizer=optimizer,
    num_epochs=settings.EPOCHS,
    threshold=settings.THRESHOLD,
    avg_results=True
)
# Result Plot of Loss and Metrics
utils.result_plot(loss_and_metrics)
