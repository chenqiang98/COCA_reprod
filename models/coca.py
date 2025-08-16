import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.resnet import resnet18, resnet50
from models.vit import vit_base_patch16_224, mobilevit_s

def get_model(model_name, pretrained=True):
    if model_name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif model_name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif model_name == 'vit_base_patch16_224':
        return vit_base_patch16_224(pretrained=pretrained)
    elif model_name == 'mobilevit_s':
        return mobilevit_s(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not found.")

class COCA(nn.Module):
    def __init__(self, anchor_model, aux_model, lr_anchor=0.00025, lr_aux=0.001, momentum=0.9,
                 include_batchnorm=True, include_layernorm=True, include_groupnorm=False, include_instancenorm=False):
        super(COCA, self).__init__()
        self.anchor_model = anchor_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.aux_model = aux_model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Norm config
        self._norm_cfg = dict(
            include_batchnorm=include_batchnorm,
            include_layernorm=include_layernorm,
            include_groupnorm=include_groupnorm,
            include_instancenorm=include_instancenorm,
    )

        # Setup optimizers for normalization layers
        self.optimizer_anchor = self.setup_optimizer(self.anchor_model, lr_anchor, momentum)
        self.optimizer_aux = self.setup_optimizer(self.aux_model, lr_aux, momentum)

    # Learnable scaling factor tau
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tau = nn.Parameter(torch.tensor(1.0, requires_grad=True, device=device))
        self.optimizer_tau = optim.SGD([self.tau], lr=0.01, momentum=momentum)

    def setup_optimizer(self, model, lr, momentum):
        # collect all trainable normalization layer parameters
        norm_params = []
        for name, module in model.named_modules():
            if self._norm_cfg['include_batchnorm'] and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for _, param in module.named_parameters():
                    if param.requires_grad:
                        norm_params.append(param)
            if self._norm_cfg['include_layernorm'] and isinstance(module, nn.LayerNorm):
                for _, param in module.named_parameters():
                    if param.requires_grad:
                        norm_params.append(param)
            if self._norm_cfg['include_groupnorm'] and isinstance(module, nn.GroupNorm):
                for _, param in module.named_parameters():
                    if param.requires_grad:
                        norm_params.append(param)
            if self._norm_cfg['include_instancenorm'] and isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                for _, param in module.named_parameters():
                    if param.requires_grad:
                        norm_params.append(param)

        # check existence of normalization layer parameters
        if not norm_params:
            # No norm parameters to adapt for this model
            return None

        # make sure normalization layer parameters are on the right device
        device = next(model.parameters()).device
        norm_params = [p.to(device) for p in norm_params]

        return optim.SGD(norm_params, lr=lr, momentum=momentum)
        # bn_params = []
        # for module in model.modules():
        #     if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        #         bn_params.extend([p for p in module.parameters() if p.requires_grad])
        # return optim.SGD(bn_params, lr=lr, momentum=momentum)

    def forward(self, x_anchor, x_aux):
        p_a = self.anchor_model(x_anchor)
        p_s = self.aux_model(x_aux)
        
        p_e_prime = p_a + (p_s / self.tau.detach())
        
        # Adaptive balance factor T with numerical stability
        max_p_e_prime = torch.max(p_e_prime, dim=1, keepdim=True)[0]
        max_p_a = torch.max(p_a, dim=1, keepdim=True)[0]
        T = max_p_e_prime / torch.clamp(max_p_a, min=1e-8)
        p_e = p_e_prime / torch.clamp(T, min=1e-8)
        
        return p_e

    def update(self, x_anchor, x_aux, debug=False):
        # Ensure models are in training mode for adaptation
        self.anchor_model.train()
        self.aux_model.train()
        
        # Forward pass
        p_a = self.anchor_model(x_anchor)
        p_s = self.aux_model(x_aux)

        # 1. Update tau
        self.optimizer_tau.zero_grad()
        l_s = torch.norm(torch.exp(p_a.detach()) - torch.exp(p_s / self.tau), p=1)
        l_s.backward()
        self.optimizer_tau.step()
        
        # Clamp tau to be positive and reasonable range
        self.tau.data.clamp_(min=0.1, max=10.0)
        
        if debug:
            print(f"Tau: {self.tau.item():.4f}, L_s: {l_s.item():.4f}")

        # 2. Form ensemble prediction
        p_e_prime = p_a + (p_s / self.tau.detach())
        
        # Adaptive balance factor T with numerical stability
        max_p_e_prime = torch.max(p_e_prime, dim=1, keepdim=True)[0].detach()
        max_p_a = torch.max(p_a, dim=1, keepdim=True)[0].detach()
        T = max_p_e_prime / torch.clamp(max_p_a, min=1e-8)
        p_e = p_e_prime / torch.clamp(T, min=1e-8)

        # 3. Calculate losses
        # Marginal entropy loss
        l_mar = self.entropy_loss(p_e)

        # Cross-model knowledge distillation loss
        y_hat = p_e.detach().argmax(dim=1)
        l_ckd_a = F.cross_entropy(p_a, y_hat)
        l_ckd_s = F.cross_entropy(p_s, y_hat)
        l_ckd = l_ckd_a + l_ckd_s

        # Self-adaptation loss
        l_self_a = self.entropy_loss(p_a)
        l_self_s = self.entropy_loss(p_s)
        l_sa = l_self_a + l_self_s

        # Total loss
        loss = l_mar + l_ckd + l_sa

        # Update models
        if self.optimizer_anchor is not None:
            self.optimizer_anchor.zero_grad()
        if self.optimizer_aux is not None:
            self.optimizer_aux.zero_grad()
        loss.backward()
        if self.optimizer_anchor is not None:
            self.optimizer_anchor.step()
        if self.optimizer_aux is not None:
            self.optimizer_aux.step()

    def entropy_loss(self, logits):
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        return -(p * log_p).sum(dim=1).mean()

