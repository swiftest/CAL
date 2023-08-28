import torch
import numpy as np
from Utils import safe_norm
from einops import rearrange
from caps_vit import CapsViT, LossNet
import torch.utils.data as Data
from auxiliary_classifier import S3KAIResNet, CapsGLOM


def initialize_models(band, patch_size, num_classes):
    target = CapsViT(band, patch_size, caps2_caps=num_classes)
    target = target.cuda()
    loss_module = LossNet()
    loss_module = loss_module.cuda()
    model_cap = {'backbone': target, 'module': loss_module}
    
    model_ask = S3KAIResNet(band, num_classes=num_classes)
    model_ask = model_ask.cuda()
    
    model_glom = CapsGLOM(band, patch_size, num_classes=num_classes)
    model_glom = model_glom.cuda()
    
    return model_cap, model_ask, model_glom


def search_informative_samples(model_cap, model_ask, model_glom, candidate_loader):
    model_cap['backbone'].eval()
    model_cap['module'].eval()
    model_ask.eval()
    model_glom.eval()
    
    candidate_uncertainty = np.array([])
    candidate_pred_cap = np.array([]).astype('int')  # to be used later
    candidate_pred_ask = np.array([]).astype('int')  # to be used later
    candidate_pred_glom = np.array([]).astype('int')  # to be used later
    tar = np.array([]).astype('int')
    collect_candidate_samples = []
    
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(candidate_loader):
            batch_data = batch_data.cuda()
            
            ask_batch = rearrange(batch_data, 'b c h w -> b h w c').unsqueeze(1)
            batch_pred_ask = model_ask(ask_batch)
            _, pred_ask = batch_pred_ask.topk(1, axis=1)
            candidate_pred_ask = np.append(candidate_pred_ask, pred_ask.squeeze().cpu().numpy())

            batch_pred_glom = model_glom(batch_data)
            _, pred_glom = batch_pred_glom.topk(1, axis=1)
            candidate_pred_glom = np.append(candidate_pred_glom, pred_glom.squeeze().cpu().numpy())
            
            batch_pred_cap, _, features = model_cap['backbone'](batch_data)
            batch_pred_cap = safe_norm(batch_pred_cap, axis=-1)  # (batch, 13)
            _, pred_cap = batch_pred_cap.topk(1, axis=1)  # (100, 1)
            candidate_pred_cap = np.append(candidate_pred_cap, pred_cap.squeeze().cpu().numpy())
            
            pred_loss = model_cap['module'](features)
            candidate_uncertainty = np.append(candidate_uncertainty, pred_loss.squeeze().cpu().numpy())  # (batch, )
            
            tar = np.append(tar, batch_target.numpy())
            collect_candidate_samples.append(batch_data.cpu())
    
    candidate_samples = torch.cat(collect_candidate_samples)
    num_candidate_sampls = len(candidate_uncertainty)
    candidate_labels = torch.from_numpy(tar)
    num_classes = np.max(tar) + 1
    ind_uncertainty = np.argsort(candidate_uncertainty)[::-1]  # Sort prediction loss from largest to smallest
    
    collect_labels = []
    collect_samples = []
    collect_index = np.array([]).astype('int')
    print("**************************************************")
    clustering_pred = {keys: [] for keys in range(num_classes)}
    for i in range(num_candidate_sampls):
        index = ind_uncertainty[i]
        pred_label_glom = candidate_pred_glom[index]
        clustering_pred[pred_label_glom].append(index)
    for j in range(num_classes):
        target_index = clustering_pred[j][0]
        collect_labels.append(tar[target_index])
        collect_samples.append(candidate_samples[target_index])
        collect_index = np.append(collect_index, target_index)    
        print("{:.10f} \t Pred Label:{:2d} \t True Label:{:2d}".format(candidate_uncertainty[target_index], j, tar[target_index]))
    print("**************************************************")
    
    informative_samples = torch.stack(collect_samples)
    informative_labels = torch.from_numpy(np.array(collect_labels))
    
    new_x_candidate = np.delete(candidate_samples, collect_index, axis=0)
    print("New_Candidate_Pool:", new_x_candidate.shape)
    new_y_candidate = np.delete(candidate_labels, collect_index, axis=0)
    print("**************************************************")
    New_Candidate_Label = Data.TensorDataset(new_x_candidate, new_y_candidate)
    new_candidate_loader = Data.DataLoader(New_Candidate_Label, batch_size=candidate_loader.batch_size, shuffle=True)
    return informative_samples, informative_labels, new_candidate_loader