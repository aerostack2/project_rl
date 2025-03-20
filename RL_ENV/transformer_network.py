import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Tuple, Type


class TransformerExtractor(nn.Module):
    """
    Transformer-based architecture that processes a variable number of candidate features
    (e.g., frontier point features) and outputs:
      - pointer_logits: unnormalized scores for each candidate (to be turned into a probability distribution)
      - latent_value: a latent representation for the critic (value function)

    The input is a tensor of shape (batch_size, num_candidates, candidate_feature_dim).
    """

    def __init__(
        self,
        candidate_feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        num_heads: int = 4,
        actor_num_layers: int = 2,
        critic_num_layers: int = 2,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super(TransformerExtractor, self).__init__()
        device = th.device(device)
        self.device = device

        # Parse the network architecture
        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])
            vf_layers_dims = net_arch.get("vf", [])
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        # ---------------------------
        # Actor network (Transformer-based Pointer)
        # ---------------------------
        actor_embedding_dim = pi_layers_dims[0] if len(
            pi_layers_dims) > 0 else candidate_feature_dim
        self.actor_embed = nn.Linear(candidate_feature_dim, actor_embedding_dim)

        # Transformer encoder for actor with batch_first=True
        encoder_layer_actor = nn.TransformerEncoderLayer(
            d_model=actor_embedding_dim,
            nhead=num_heads,
            batch_first=True,
            activation=activation_fn().__class__.__name__.lower()  # e.g., "relu"
        )
        self.transformer_actor = nn.TransformerEncoder(
            encoder_layer_actor, num_layers=actor_num_layers)

        # Additional actor MLP layers (if any) after the Transformer encoder
        actor_mlp_layers = []
        last_dim = actor_embedding_dim
        for layer_dim in pi_layers_dims[1:]:
            actor_mlp_layers.append(nn.Linear(last_dim, layer_dim))
            actor_mlp_layers.append(activation_fn())
            last_dim = layer_dim
        self.actor_mlp = nn.Sequential(*actor_mlp_layers) if actor_mlp_layers else nn.Identity()

        # Final linear layer to produce a score for each candidate
        self.actor_score = nn.Linear(last_dim, 1)

        # ---------------------------
        # Critic network (Transformer-based)
        # ---------------------------
        critic_embedding_dim = vf_layers_dims[0] if len(
            vf_layers_dims) > 0 else candidate_feature_dim
        self.critic_embed = nn.Linear(candidate_feature_dim, critic_embedding_dim)
        encoder_layer_critic = nn.TransformerEncoderLayer(
            d_model=critic_embedding_dim,
            nhead=num_heads,
            batch_first=True,
            activation=activation_fn().__class__.__name__.lower()
        )
        self.transformer_critic = nn.TransformerEncoder(
            encoder_layer_critic, num_layers=critic_num_layers)

        critic_mlp_layers = []
        last_critic_dim = critic_embedding_dim
        for layer_dim in vf_layers_dims[1:]:
            critic_mlp_layers.append(nn.Linear(last_critic_dim, layer_dim))
            critic_mlp_layers.append(activation_fn())
            last_critic_dim = layer_dim
        self.critic_mlp = nn.Sequential(*critic_mlp_layers) if critic_mlp_layers else nn.Identity()

        # Save the latent dimension for the critic output (for use in a value head later)
        self.latent_dim_vf = last_critic_dim

    def forward(self, candidate_features: List[th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        """
        :param candidate_features: A tensor of shape 
            (batch_size, num_candidates, candidate_feature_dim)
        :return: A tuple (pointer_logits, latent_value) where:
            - pointer_logits: Tensor of shape (batch_size, num_candidates) with unnormalized scores.
            - latent_value: Tensor of shape (batch_size, latent_dim_vf) for the critic.
        """
        pointer_logits = self.forward_actor(candidate_features)
        latent_value = self.forward_critic(candidate_features)
        return pointer_logits, latent_value

    def forward_actor(self, candidate_features: List[th.Tensor]) -> th.Tensor:
        # candidate_features: (B, num_candidates, candidate_feature_dim)
        # (B, num_candidates, actor_embedding_dim)
        pointer_logits_list = []

        for candidate_feature in candidate_features:
            actor_emb = self.actor_embed(candidate_feature)
            # (B, num_candidates, actor_embedding_dim)
            transformer_out = self.transformer_actor(actor_emb)
            actor_out = self.actor_mlp(transformer_out)  # (B, num_candidates, last_dim)
            scores = self.actor_score(actor_out)  # (B, num_candidates, 1)
            pointer_logits = scores.squeeze(-1)  # (B, num_candidates)

            pointer_logits_list.append(pointer_logits)

        pointer_logits = th.cat(pointer_logits_list, dim=0)  # (B, num_candidates)

        return pointer_logits

    def forward_critic(self, candidate_features: List[th.Tensor]) -> th.Tensor:
        # candidate_features: (B, num_candidates, candidate_feature_dim)
        # (B, num_candidates, critic_embedding_dim)
        latent_values_list = []

        for candidate_feature in candidate_features:
            critic_emb = self.critic_embed(candidate_feature)
            # (B, num_candidates, critic_embedding_dim)
            transformer_out = self.transformer_critic(critic_emb)
            # Pool over candidates (using mean pooling here)
            pooled = transformer_out.mean(dim=1)  # (B, critic_embedding_dim)
            latent_value = self.critic_mlp(pooled)  # (B, latent_dim_vf)

            latent_values_list.append(latent_value)

        latent_value = th.cat(latent_values_list, dim=0)  # (B, latent_dim_vf)

        return latent_value
