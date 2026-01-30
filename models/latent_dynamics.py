import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from utils.tf_utils import add_twist_to_pose
from models.support_models import AdaptiveLayerNorm, SinusoidalPosEmb, TransformerBlock


# Flow matching transformer dynamics model
class SimpleFlowMatchingDynamics(nn.Module):
    """
    Latent state dynamics model. This model takes the PC/one-hot latent state of the object, the action, the flow matching phase,
    and the sampled state at the current denoising step and outputs the change in latent state (relative latent state) for the next
    phase of the flow matching process. The flow model itself is a transformer encoder.
    Args:
        max_objects (int): maximum number of objects in the scene   (max_seq_len = 2*max_objects + 1)
        width (int): dimension of the latent state embedding        (hidden dim)
        n_layers (int): number of transformer encoder layers        (num_layers)
        n_heads (int): number of attention heads in the transformer encoder
        action_size (int): dimension of the action input
        dim_feedforward (int): dimension of the feedforward network in the transformer encoder
        skill (str): skill type for dynamics model (unused currently)
        dynamics_type (str): type of dynamics model (e.g., 'transformer')
    """
    def __init__(self, max_objects: int, width: int, 
                 n_layers: int, n_heads: int, 
                 action_size: int, dim_feedforward: int,
                 skill='push', dynamics_type='transformer', 
                 phase_dim: int=256, mlp_ratio: float=4.0, 
                 dropout: float=0.0, input_dim: int=9,
                 output_dim: int=6
        ):
        super(SimpleFlowMatchingDynamics, self).__init__()

        self.max_objects = max_objects
        emb_dim = (int)(width/2) # want this to be 256 / 2 = 128

        # check that the width is divisible by 2
        assert width % 2 == 0, "Width must be divisible by 2"

        ################ Contextual Embeddings ################
        # one_hot embedding for object ids: -> [B, 128]
        self.emb_onehot =  nn.Embedding(max_objects+1, emb_dim)
        
        # TODO different mlp for each skill label
        # continuous action embedding:  -> [B, 128]
        self.emb_act_continous = nn.Sequential(
            nn.Linear(action_size, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        ################ Latent Flow Decoder ################
        # This decouples the "Geometric Twist" prediction from the "Semantic State" prediction
        self.latent_flow_proj = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width) # Output same dim as zs
        )

        ################ Input Projection ################
        # (9D pose: pos+ortho6d): [B, M, 9] -> [B, M, 256]
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, width)
        )

        ################ Positional Embeddings ################
        # Sequence contains: 1 action (za) + M noisy objects (z_phase)
        max_seq_len = max_objects + 1
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, width))

        ################ Phase Embedding ################
        # phase embedding (converts scalar phase to embedding), (1 dim): [B, 1] -> [B, 256]
        self.phase_emb = nn.Sequential(
            SinusoidalPosEmb(phase_dim),
            nn.Linear(phase_dim, phase_dim),
            nn.SiLU(),
            nn.Linear(phase_dim, phase_dim)
        )

        ################ Transformer blocks ################
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=width,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                phase_dim=phase_dim
            )
            for _ in range(n_layers)
        ])

        ################ Output decoder ################
        # decoder: [B, M, 256] -> [B, M, 6]
        self.final_norm = AdaptiveLayerNorm(width, phase_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(width, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, output_dim),
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)
        nn.init.normal_(self.pos_emb, std=0.02)

        # Zero-initialize AdaLN modulation layers for stability (DiT/SiT Best Practice)
        # This makes the transformer blocks act as identity functions at init
        for block in self.blocks:
            nn.init.constant_(block.norm1.phase_proj[-1].weight, 0)
            nn.init.constant_(block.norm1.phase_proj[-1].bias, 0)
            nn.init.constant_(block.norm2.phase_proj[-1].weight, 0)
            nn.init.constant_(block.norm2.phase_proj[-1].bias, 0)
        
        # Zero-initialize output projection for better training stability
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
                
    def forward(self, a:Tensor, phase:Tensor, x_phase:Tensor, src_key_padding_mask=None, is_action_embedded:bool=False) -> Tensor:
        """
        B: batch size, M: max number of objects
        Args:
            a (Tensor): A raw action at the current timestep t (B,M+18) OR embedded action (B, width)
            phase (Tensor): flow-matching phase parameter [B, 1]
            x_phase (Tensor): pose at flow-matching phase (pos+ortho6s) [B, M, 9]
            src_key_padding_mask (Tensor): A padding mask for transformer network
            is_action_embedded (bool): Whether 'a' is already a latent embedding

        Returns:
            delta_z_dot (Tensor): predicted latent velocity (B, M, width)
            next_z (Tensor): predicted next latent state (B, M, width)
        """
        B = a.shape[0]
        M = self.max_objects
        
        if is_action_embedded:
            za = a
            if za.dim() == 2:
                za = za.unsqueeze(1)  # (B, 1, 256)
        else:
            # Embed action
            za1 = self.emb_onehot(torch.argmax(a[:,:M], dim=-1)) # (B, 128)
            za1 = za1.reshape(B, 1, za1.shape[1]) # (B, 1, 128)
            za2 = self.emb_act_continous(a[:,M:]) # (B, 128)
            za2 = za2.reshape(B, 1, za2.shape[1]) # (B, 1, 128)
            za = torch.cat((za1, za2), dim=-1) # (B, 1, 256)

        # Ensure phase is the right shape
        if phase.dim() == 1:
            phase = phase.view(-1)
        elif phase.dim() == 2:
            phase = phase.squeeze(-1)
        phase_emb = self.phase_emb(phase)  # (B, phase_dim)

        # project input pose to latent space
        z_phase = self.input_proj(x_phase) # (B, M, 256)

        # concat object embeddings, action embedding, pose embeddings, and position embeddings
        z = torch.cat((za, z_phase), dim=1) # (B, 1+M, 256)

        # add positional embeddings
        z = z + self.pos_emb[:, :z.size(1), :]

        # Apply transformer blocks with phase conditioning
        for block in self.blocks:
            z = block(z, phase_emb)

        # Extract the output corresponding to the denoising phase
        z_dot = z[:, 1:, :]  # (B, M, width)

        # 1. Output GEOMETRIC Flow (Twist)
        # We process z_dot for the pose decoder
        x_dot = self.output_proj(self.final_norm(z_dot, phase_emb))  # (B, M, 6)
        
        # Return:
        # x_dot -> Goes to Velocity Loss (Flow Matching)
        return x_dot

    @torch.no_grad()
    def sample(self, zs, a, x0, src_key_padding_mask=None, num_steps=100, method='euler', is_action_embedded:bool=False):
        """
        Generate samples using ODE integration.
        
        Args:
            zs (Tensor): A latent state embedding of object states at the current timestep t (B,M,emb_dim)
            a (Tensor): A raw action at the current timestep t (B,M+18) OR embedded action (B, width)
            x0: initial noise samples [batch, M, input_dim]
            num_steps: number of integration steps
            method: integration method ('euler' or 'midpoint')
            is_action_embedded (bool): Whether 'a' is pre-embedded
            
        Returns:
            x1: generated samples [batch, seq_len, input_dim]
        """
        device = x0.device
        B = x0.shape[0]
        
        x = x0.clone()
        dt = torch.tensor(1.0 / num_steps, device=device)
        
        for step in range(num_steps):
            t = torch.full((B,), step * dt.item(), device=device)
            
            # Euler method
            # TODO: remove [0] if returning x_dot only
            v = self.forward(zs, a, t, x, src_key_padding_mask, is_action_embedded=is_action_embedded)[0]
            x = add_twist_to_pose(x, v, dt)
        
        return x


