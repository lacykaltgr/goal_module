import torch
from pytorch_lightning import LightningModule
from models.gmc_networks import *


class GMC(LightningModule):
    def __init__(self, name, common_dim, latent_dim):
        super(GMC, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.image_processor = None
        self.label_processor = None
        self.joint_processor = None
        self.processors = [
            self.image_processor,
            self.label_processor,
            self.joint_processor,
        ]

        self.encoder = None

    def encode(self, x, sample=False):

        # If we have complete observations
        if None not in x:
            return self.encoder(self.processors[-1](x))
        else:
            latent_representations = []
            for id_mod in range(len(x)):
                if x[id_mod] is not None:
                    latent_representations.append(self.encoder(self.processors[id_mod](x[id_mod])))

            # Take the average of the latent representations
            latent = torch.stack(latent_representations, dim=0).mean(0)
            return latent

    def forward(self, x):
        # Forward pass through the modality specific encoders
        batch_representations = []
        for processor_idx in range(len(self.processors) - 1):
            mod_representations = self.encoder(
                self.processors[processor_idx](x[processor_idx])
            )
            batch_representations.append(mod_representations)

        # Forward pass through the joint encoder
        joint_representation = self.encoder(self.processors[-1](x))
        batch_representations.append(joint_representation)
        return batch_representations

    def infonce(self, batch_representations, temperature, batch_size):
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict

    def training_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Compute contrastive loss
        loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return loss, tqdm_dict

    def validation_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)
        # Compute contrastive loss
        loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return tqdm_dict


class DummyGoalGMC(GMC):
    def __init__(self, name, common_dim, latent_dim, loss_type):
        super(DummyGoalGMC, self).__init__(name, common_dim, latent_dim)

        self.features_processor = ImageFeaturePairProcessor(common_dim=common_dim)
        self.coord_processor = CoordProcessor(common_dim=common_dim)
        self.joint_processor = JointProcessor(common_dim=common_dim)
        self.processors = [
            self.features_processor,
            self.coord_processor,
            self.joint_processor,
        ]
        self.loss_type = loss_type

        self.encoder = SharedProjection(common_dim=common_dim, latent_dim=latent_dim)