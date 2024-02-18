import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.pixel_wise_mapping import remap_using_flow_fields
from utils_training.multiscale_loss import *
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from imageio import imread
import torchvision.transforms as tf
import os
from torchnet.meter.confusionmeter import ConfusionMeter
from utils_training.preprocess_batch import pre_process_change,pre_process_data
from utils.plot import overlay_result
from utils.evaluate import IoU
from pytorch_msssim import ms_ssim, ssim
import flow_vis
import cv2
from models.our_models.mod import warp
from datasets.changesim import SegHelper
import kornia
from torchvision.transforms.functional import rgb_to_grayscale


def edge_detect(img, sigma=0.33):
    # gray_img = rgb_to_grayscale(img)
    
    # v = torch.median(gray_img).item()
    # lower = max(0, (1.0-sigma)*v)
    # upper = min(1, (1.0+sigma)*v)
    # _, edge_map = kornia.filters.canny(gray_img, low_threshold=lower, high_threshold=upper, hysteresis=False)
    _, edge_map = kornia.filters.canny(img, low_threshold=0.2, high_threshold=0.99, hysteresis=False)
    
    return edge_map

def mask_to_png(mask, colormap, img):
    """
    Args:
        mask: [h, w], np.array, uint8.
        colormap: [n, 2], np.array.
        img: [h, w, 3], np.array, uint8.
    """
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask[mask == 255] = 0

    for l in np.unique(mask):
        rgb_mask[mask == l] = colormap[int(l)]

    overlay = cv2.addWeighted(img, 1.0, rgb_mask, 0.7,0)

    return rgb_mask, overlay   

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def denormalize(img):
    mean_vector = np.array([0.485, 0.456, 0.406])
    std_vector = np.array([0.229, 0.224, 0.225])
    mean = torch.as_tensor(mean_vector, dtype=img.dtype, device=img.device)
    std = torch.as_tensor(std_vector, dtype=img.dtype, device=img.device)
    
    img_copy = img.mul(std[:, None, None]).add(mean[:, None, None])
    
    return img_copy

def calc_flow_std(flow, patch_size=16, patch_stride=16):
    # flow: B 2 H W
    flow_patches = flow.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
    b, c, num_patch_h, num_patch_w, patch_h, patch_w = flow_patches.shape
    flow_patches = flow_patches.reshape(b, c, num_patch_h, num_patch_w, patch_h * patch_w)
    std_map = flow_patches.std(dim=4).mean(dim=1)
    flow_patches = flow_patches.reshape(b, c, num_patch_h * num_patch_w, patch_h * patch_w)

    flow_stds = flow_patches.std(dim=3).mean(dim=2).mean(dim=1)
    return flow_stds, std_map

def fused_feat_sim(feats_src, feats_tgt, flow, h_img, hw=None, mode='relu'):
    num_feats = len(feats_src)
    feat_sim_list = []
    h_f, w_f = flow.size()[-2], flow.size()[-1] if hw is None else hw
    vmask = warp(None, flow*(h_f / h_img))
    for f_src, f_tgt in zip(feats_src, feats_tgt):
        _, _, h, w = f_src.size()
        div_factor = h / h_img  # for scale flow1 value
        flow_ds = F.interpolate(flow, size=(h, w), mode='bilinear')
        warped_f_src = warp(f_src, flow_ds*div_factor)
        cos_sim = F.cosine_similarity(warped_f_src, f_tgt, dim=1).unsqueeze(1)
        # non-negative similarity
        # negative correlations are treated as orthogonal       
        feat_sim = F.interpolate(cos_sim, size=(h_f, w_f), mode='bilinear')
        
        if mode == 'relu':
            feat_sim_list.append(F.relu(feat_sim))
        else:
            feat_sim_list.append((1.0+feat_sim)/2.0)
        
    k = len(feats_src) if mode == 'relu' else 1
    fused_feat_sim = (torch.prod(torch.stack(feat_sim_list), dim=0)).pow(1/k)
        
    return fused_feat_sim, vmask.unsqueeze(1)

def fused_feat_diff(feats_src, feats_tgt, flow, h_img, hw=None):
    num_feats = len(feats_src)
    feat_diff_list = []
    h_f, w_f = flow.size()[-2], flow.size()[-1] if hw is None else hw
    vmask = warp(None, flow*(h_f / h_img))
    for f_src, f_tgt in zip(feats_src, feats_tgt):
        _, _, h, w = f_src.size()
        div_factor = h / h_img  # for scale flow1 value
        flow_ds = F.interpolate(flow, size=(h, w), mode='bilinear')
        warped_f_src = warp(f_src, flow_ds*div_factor)
        l1_dist = (warped_f_src - f_tgt).abs().sum(dim=1, keepdim=True)
        # non-negative similarity
        # negative correlations are treated as orthogonal
        feat_diff = F.interpolate(l1_dist, size=(h_f, w_f), mode='bilinear')
        # feat_sim_list.append(F.relu(feat_sim))
        feat_diff_list.append(feat_diff)
    
    fused_feat_diff = (torch.prod(torch.stack(feat_diff_list), dim=0))
        
    return fused_feat_diff, vmask.unsqueeze(1)


def plot_during_training2(save_path, epoch, batch, apply_mask,
                         h_original, w_original, h_256, w_256,
                         source_image, target_image, source_image_256, target_image_256, div_flow,
                         flow_gt_original, flow_gt_256, output_net,  output_net_256,
                         target_change_original,
                         target_change_256,
                         out_change_orig,
                         out_change_256,
                         mask=None, mask_256=None,
                         f_sim_map=None, vmask=None,
                         return_img = False,
                         save_split=False,
                         seg_helper = SegHelper(),
                         multi_class=1,
                         tr_type='',
                         src_edge=None, tgt_edge=None,
                         make_vid=False):
    # resolution original
    flow_est_original = F.interpolate(output_net, (h_original, w_original),
                                      mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_target_x = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)
    image_1 = (source_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2 = (target_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    
    if src_edge is not None:
        # src_edge = F.interpolate(src_edge,(h_original, w_original),
        #                         mode='nearest')
        # tgt_edge = F.interpolate(tgt_edge,(h_original, w_original),
        #                         mode='nearest')
        
        src_edge = (src_edge[0].cpu().permute(1, 2, 0))
        tgt_edge = (tgt_edge[0].cpu().permute(1, 2, 0))
    
    # remapped_gt = remap_using_flow_fields(image_1.numpy(),
    #                                       flow_target_x.cpu().numpy(),
    #                                       flow_target_y.cpu().numpy())
    
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())
    if target_change_original is not None:
        target_change_original = 50*target_change_original[0][0].cpu()
    out_change_original = F.interpolate(out_change_orig[-1],(h_original, w_original),
                                        mode='bilinear',align_corners=False)
    vmask = F.interpolate(vmask.float(), size=(h_original, w_original), mode='bilinear', align_corners=False).bool()
    if multi_class > 1:
        out_change_original = out_change_original[0].argmax(0)
        out_change_label = out_change_original.cpu().numpy()
    else:
        out_change_prob = torch.sigmoid(out_change_original[0])
        out_change_label = (out_change_prob.ge(0.5).long()*vmask[0]).squeeze(0)
        out_change_label = 50*out_change_label.cpu().numpy()
        out_change_prob = out_change_prob.permute(1, 2, 0).cpu().numpy()
    
    f_sim_map = F.interpolate(f_sim_map, size=(h_original, w_original),
                              mode='bilinear', align_corners=False).permute(0, 2, 3, 1)[0].cpu().numpy()

    if not make_vid:
        if save_split:
            if not os.path.isdir(os.path.join(save_path,'t0')): os.mkdir(os.path.join(save_path,'t0'))
            if not os.path.isdir(os.path.join(save_path,'t1')): os.mkdir(os.path.join(save_path,'t1'))
            if not os.path.isdir(os.path.join(save_path,'pred_on_t1')): os.mkdir(os.path.join(save_path,'pred_on_t1'))
            if not os.path.isdir(os.path.join(save_path,'pred_on_remapped')): os.mkdir(os.path.join(save_path,'pred_on_remapped'))
            if not os.path.isdir(os.path.join(save_path,'gt_on_t1')): os.mkdir(os.path.join(save_path,'gt_on_t1'))
            if not os.path.isdir(os.path.join(save_path,'flow')): os.mkdir(os.path.join(save_path,'flow'))
            if not os.path.isdir(os.path.join(save_path,'uncertainty')): os.mkdir(os.path.join(save_path,'uncertainty'))

            # temp viz start

            flow_gt1 = div_flow * flow_est_original[0].permute(1, 2, 0).detach().cpu().numpy()  # now shape is HxWx2
            flow_gt1 = cv2.resize(flow_vis.flow_to_color(flow_gt1), dsize=(640,480), interpolation=cv2.INTER_CUBIC)
            plt.imsave('{}/flow/epoch{}_batch{}.png'.format(save_path, epoch, batch), flow_gt1)
            flow_stds, std_map1 = calc_flow_std(flow_est_original)
            std_map1 = std_map1[0]
            std_map1 = std_map1.detach().cpu().clamp(min=0.0, max=5.0).numpy()
            std_map1 = cv2.resize(std_map1,dsize=(640,480),interpolation=cv2.INTER_NEAREST)
            plt.imsave('{}/uncertainty/epoch{}_batch{}.png'.format(save_path, epoch, batch),std_map1 )
            out_change_original1 = out_change_original[:,:,None].astype(np.bool8)
            out_change1 = overlay_result(out_change_original1,image_2.numpy(),color=None)
            out_change_remapped1 = overlay_result(out_change_original1,remapped_est, color=None)
            out_change1 = cv2.resize(out_change1,dsize=(640,480),interpolation=cv2.INTER_LINEAR)
            out_change_remapped1 = cv2.resize(out_change_remapped1,dsize=(640,480),interpolation=cv2.INTER_LINEAR)
            plt.imsave('{}/pred_on_t1/epoch{}_batch{}.png'.format(save_path, epoch, batch),out_change1)
            plt.imsave('{}/pred_on_remapped/epoch{}_batch{}.png'.format(save_path, epoch, batch),out_change_remapped1)
            image_1_ = cv2.resize(image_1.numpy(),dsize=(640,480),interpolation=cv2.INTER_LINEAR)
            image_2_ = cv2.resize(image_2.numpy(),dsize=(640,480),interpolation=cv2.INTER_LINEAR)
            plt.imsave('{}/t0/epoch{}_batch{}.png'.format(save_path, epoch, batch),image_1_)
            plt.imsave('{}/t1/epoch{}_batch{}.png'.format(save_path, epoch, batch),image_2_)

            if return_img:
                return target_change_original.transpose(2,0,1)


        else:
            num_figs=8 if target_change_original is not None else 7
            fig, axis = plt.subplots(1, num_figs, figsize=(20, 10))
            axis[0].imshow(image_1.numpy())
            axis[0].set_title("Ref. Image")
            axis[0].axis('off')
            axis[1].imshow(image_2.numpy())
            axis[1].set_title("Query Image")
            axis[1].axis('off')

            # if apply_mask:
            #     mask = mask.detach()[0].cpu().numpy().astype(np.float32)
            # else:
            #     mask = np.ones((h_original, w_original))

            flow_stds, std_map = calc_flow_std(flow_est_original)
            flow_stds = flow_stds[0]
            std_map = std_map[0]

            # flow_gt = div_flow * flow_est_original[0].permute(1, 2, 0).detach().cpu().numpy()  # now shape is HxWx2
            # axis[0,2].imshow(flow_vis.flow_to_color(flow_gt))
            # axis[0,2].set_title('Flow')
            # axis[0,2].axis('off')

            # std_map = axis[0,3].imshow(std_map.detach().cpu().clamp(min=0.0,max=5.0).numpy())
            # # fig.colorbar(std_map)
            # axis[0,3].set_title("Uncertainty (score={:.2f})".format(flow_stds))
            # axis[0,3].axis('off')

            out_change_original_overlayed = overlay_result(out_change_label[:,:,None].astype(np.bool8),image_2.numpy(),color=None)
            axis[4].imshow(out_change_original_overlayed,vmax=255,interpolation='nearest')
            axis[4].set_title("Estim. on Query")
            axis[4].axis('off')
            
            if src_edge is not None:
                axis[2].imshow(tgt_edge.numpy(), cmap='jet')
                axis[2].set_title("Query Edge")
                axis[2].axis('off')
                axis[3].imshow(src_edge.numpy(), cmap='jet')
                axis[3].set_title("Ref. Image")
                axis[3].axis('off')        
            
            remapped_est = overlay_result(out_change_label[:, :, None].astype(np.bool8),
                                                    remapped_est, color=None)
            axis[5].imshow(remapped_est)
            axis[5].set_title("Estim. on Warped Ref.")
            axis[5].axis('off')
        
            axis[6].imshow(f_sim_map, cmap='jet')
            axis[6].set_title("Pseudo Change")
            axis[6].axis('off')
            
            # axis[1,5].imshow(tgt_edge.numpy(), cmap='jet')
            # axis[1,5].set_title("Masked Query Edge")
            # axis[1,5].axis('off')
            
            # axis[1,6].imshow(src_edge.numpy(), cmap='jet')
            # axis[1,6].set_title("Masked Ref. Edge")
            # axis[1,6].axis('off')

            if target_change_original is not None:
                target_change_original_overlayed = overlay_result(target_change_original[:, :, None].numpy().astype(np.bool8),
                                                        image_2.numpy(), color=None)
                axis[7].imshow(target_change_original_overlayed,vmax=255,interpolation='nearest')
                axis[7].set_title("GT on Query")
                axis[7].axis('off')

            # axis[1,7].imshow(out_change_prob, cmap='jet')
            # axis[1,7].set_title("Change Prob. Map")
            # axis[1,7].axis('off')
            
            fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                        bbox_inches='tight')
            plt.close(fig)
            if return_img:
                vis_result = imread('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch)).astype(np.uint8)[:,:,:3]
                return vis_result.transpose(2,0,1) # channel first
    else:
        fig, axis = plt.subplots(2, 2, figsize=(10, 10))
        fig.tight_layout()
        axis[0, 0].imshow(image_1.numpy())
        axis[0, 0].set_title("Ref. Image")
        axis[0, 0].axis('off')
        axis[0, 1].imshow(image_2.numpy())
        axis[0, 1].set_title("Query Image")
        axis[0, 1].axis('off')
        
        if multi_class == 1:
            remapped_est = overlay_result(out_change_label[:, :, None].astype(np.bool8),
                                                    remapped_est, color=None)
            axis[1, 0].imshow(remapped_est)
            axis[1, 0].set_title("Estim. on Warped Ref.")
            axis[1, 0].axis('off')

            out_change_original_overlayed = overlay_result(out_change_label[:,:,None].astype(np.bool8),image_2.numpy(),color=None)
            axis[1, 1].imshow(out_change_original_overlayed,vmax=255,interpolation='nearest')
            axis[1, 1].set_title("Estim. on Query")
            axis[1, 1].axis('off')
        else:
            # out_change_label [H, W], 0 ~ 4
            cmap = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]]
            _, seg_r = mask_to_png(out_change_label.astype('uint8'), cmap, (255*remapped_est).astype('uint8'))
            _, seg_q = mask_to_png(out_change_label.astype('uint8'), cmap, (255*image_2.numpy()).astype('uint8'))

            axis[1, 0].imshow(seg_r)
            axis[1, 0].set_title("Estim. on Warped Ref.")
            axis[1, 0].axis('off')

            axis[1, 1].imshow(seg_q,vmax=255,interpolation='nearest')
            axis[1, 1].set_title("Estim. on Query")
            axis[1, 1].axis('off')

    
        
        fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                    bbox_inches='tight')
        plt.close(fig)
        if return_img:
            vis_result = imread('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch)).astype(np.uint8)[:,:,:3]
            return vis_result.transpose(2,0,1) # channel first


def train_epoch(args, net,
                optimizer,
                train_loader,
                device,
                epoch,
                writer,
                loss_grid_weights=None,
                apply_mask=False,
                robust_l1_loss=False,
                sparse=False, smooth_map='edge'):
    
    net.train()
    
    l_f = args.l_f
    l_cr = args.l_cr
    l_sm = args.l_sm
    l_s_sup = args.l_s_sup
    l_s_semi = args.l_s_semi
    alpha = args.f_alpha
    beta = args.f_beta
    t_param = args.t_param

    multi_class = args.multi_class
    
    running_total_loss = 0
    running_flow_loss = 0
    running_change_loss = 0
    running_cl_loss = 0
    running_feat_loss = 0
    running_usl_cng_mean = 0
    running_fsim_mean = 0
    running_cng_reg = 0
    running_cng_smooth = 0
    r_l_semi_flow = 0
    r_l_semi_change = 0
    running_u_s_cng_mean = 0
    
    sup_train_loader = train_loader['sup']
    sup_iter = iter(sup_train_loader)
    usl_train_loader = train_loader['usl']
    
    if usl_train_loader is None: # synthetic only
        pbar = tqdm(enumerate(sup_train_loader), total=len(sup_train_loader))
    else:
        pbar = tqdm(enumerate(usl_train_loader), total=len(usl_train_loader))

    for i, tgt_batch in pbar:
        if usl_train_loader is None:
            sup_batch = tgt_batch
        else:
            try:
                sup_batch = next(sup_iter)
            except StopIteration:
                sup_iter = iter(sup_train_loader)
                sup_batch = next(sup_iter)
        
        optimizer.zero_grad()
        
        # if (i <= 3) or (i >= len(usl_train_loader)-3):
        #     print(usl_batch['index'])
        #     print(sup_batch['index'])
        
        # pre-process the data
        sup_src_image, sup_tgt_image, sup_src_image_256, sup_tgt_image_256 = pre_process_data(
            sup_batch['source_image'],
            sup_batch['target_image'],
            device=device,
            norm=args.img_norm_type)
        
        source_gt_change, target_gt_change, source_gt_change_256, target_gt_change_256 = \
            pre_process_change(sup_batch['source_change'],
            sup_batch['target_change'],
            device=device)
        use_flow = sup_batch['use_flow'][...,None].to(device)


        # output of labeled input
        sup_out_dict = net(sup_src_image, sup_tgt_image, sup_src_image_256, sup_tgt_image_256)
        sup_teacher_dict, sup_student_dict = sup_out_dict['teacher'], sup_out_dict['student']
        
        sup_t_flow_256, sup_t_flow = sup_teacher_dict['flow']
        sup_t_change_256, sup_t_change = sup_teacher_dict['change']
        
        # At original resolution
        flow_gt_original = sup_batch['flow_map'].to(device)
        
        if flow_gt_original.shape[1] != 2:
            # shape is bxhxwx2
            flow_gt_original = flow_gt_original.permute(0,3,1,2)
        bs, _, h_original, w_original = flow_gt_original.shape
        weights_original = loss_grid_weights[-len(sup_t_flow):]

        # at 256x256 resolution, b, _, 256, 256
        if sparse:
            flow_gt_256 = sparse_max_pool(flow_gt_original, (256, 256))
        else:
            flow_gt_256 = F.interpolate(flow_gt_original, (256, 256),
                                        mode='bilinear', align_corners=False)
        flow_gt_256[:,0,:,:] *= 256.0/float(w_original)
        flow_gt_256[:,1,:,:] *= 256.0/float(h_original)
        bs, _, h_256, w_256 = flow_gt_256.shape
        weights_256 = loss_grid_weights[:len(sup_t_flow_256)]
        
        loss_sup_flow = multiscaleEPE(sup_t_flow, flow_gt_original, weights=weights_original, sparse=False,
                                  mean=True, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
        loss_sup_flow += multiscaleEPE(sup_t_flow_256, flow_gt_256, weights=weights_256, sparse=False,
                                  mean=True, robust_L1_loss=robust_l1_loss,use_flow=use_flow)

        loss_sup_change = multiscaleCE(sup_t_change_256, target_gt_change_256, weights=weights_256, multi_class=multi_class)
        loss_sup_change +=multiscaleCE(sup_t_change, target_gt_change, weights=weights_original, multi_class=multi_class)
        
        loss_total = loss_sup_flow + loss_sup_change

        if args.tr_type == 'semi':
            sup_s_flow_256, sup_s_flow = sup_student_dict['flow']
            sup_s_change_256, sup_s_change = sup_student_dict['change']
            
            loss_s_sup_flow = multiscaleEPE(sup_s_flow, flow_gt_original, weights=weights_original, sparse=False,
                                            mean=True, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
            loss_s_sup_flow += multiscaleEPE(sup_s_flow_256, flow_gt_256, weights=weights_256, sparse=False,
                                             mean=True, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
            loss_s_sup_change = multiscaleCE(sup_s_change_256, target_gt_change_256, weights=weights_256, multi_class=multi_class)
            loss_s_sup_change += multiscaleCE(sup_s_change, target_gt_change, weights=weights_original, multi_class=multi_class)     
        
            loss_total += l_s_sup*(loss_s_sup_flow + loss_s_sup_change)
        
        if args.tr_type == 'sup':
            if usl_train_loader is None:
                pass
            else:
                sup_batch = tgt_batch
                
                sup_src_image, sup_tgt_image, sup_src_image_256, sup_tgt_image_256 = pre_process_data(
                    sup_batch['source_image'],
                    sup_batch['target_image'],
                    device=device,
                    norm=args.img_norm_type)
                
                source_gt_change, target_gt_change, source_gt_change_256, target_gt_change_256 = \
                    pre_process_change(sup_batch['source_change'],
                    sup_batch['target_change'],
                    device=device)
                use_flow = sup_batch['use_flow'][...,None].to(device)
                
                # output of labeled input
                sup_out_dict = net(sup_src_image, sup_tgt_image, sup_src_image_256, sup_tgt_image_256)
                sup_teacher_dict, sup_student_dict = sup_out_dict['teacher'], sup_out_dict['student']
                
                sup_t_flow_256, sup_t_flow = sup_teacher_dict['flow']
                sup_t_change_256, sup_t_change = sup_teacher_dict['change']
                
                # At original resolution
                flow_gt_original = sup_batch['flow_map'].to(device)
                
                if flow_gt_original.shape[1] != 2:
                    # shape is bxhxwx2
                    flow_gt_original = flow_gt_original.permute(0,3,1,2)
                bs, _, h_original, w_original = flow_gt_original.shape
                weights_original = loss_grid_weights[-len(sup_t_flow):]

                # at 256x256 resolution, b, _, 256, 256
                if sparse:
                    flow_gt_256 = sparse_max_pool(flow_gt_original, (256, 256))
                else:
                    flow_gt_256 = F.interpolate(flow_gt_original, (256, 256),
                                                mode='bilinear', align_corners=False)
                flow_gt_256[:,0,:,:] *= 256.0/float(w_original)
                flow_gt_256[:,1,:,:] *= 256.0/float(h_original)
                bs, _, h_256, w_256 = flow_gt_256.shape
                weights_256 = loss_grid_weights[:len(sup_t_flow_256)]
                
                loss_total += multiscaleEPE(sup_t_flow, flow_gt_original, weights=weights_original, sparse=False,
                                        mean=True, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
                loss_total += multiscaleEPE(sup_t_flow_256, flow_gt_256, weights=weights_256, sparse=False,
                                        mean=True, robust_L1_loss=robust_l1_loss,use_flow=use_flow)

                loss_total += multiscaleCE(sup_t_change_256, target_gt_change_256, weights=weights_256, multi_class=multi_class)
                loss_total +=multiscaleCE(sup_t_change, target_gt_change, weights=weights_original, multi_class=multi_class)
            
        else: # USL or Semi-SL
            # try:
            #     usl_batch = next(usl_iter)
            # except StopIteration:
            #     usl_iter = iter(usl_train_loader)
            #     usl_batch = next(usl_iter)
            
            # unlabeled images
            usl_src_image, usl_tgt_image, usl_src_image_256, usl_tgt_image_256 = pre_process_data(
            tgt_batch['source_image'],
            tgt_batch['target_image'],
            device=device,
            norm=args.img_norm_type)
            
            # augmented unlabeled images
            aug_usl_src_image, aug_usl_tgt_image, aug_usl_src_image_256, aug_usl_tgt_image_256 = pre_process_data(
            tgt_batch['aug_source_image'],
            tgt_batch['aug_target_image'],
            device=device,
            norm=args.img_norm_type)      
                        
            # output of unlabeled batch
            usl_out_dict = net(usl_src_image, usl_tgt_image, usl_src_image_256, usl_tgt_image_256,
                               aug_usl_src_image, aug_usl_tgt_image, aug_usl_src_image_256, aug_usl_tgt_image_256)
            usl_teacher_dict, usl_student_dict = usl_out_dict['teacher'], usl_out_dict['student']
            
            # split teacher and student output
            usl_t_flow_256, usl_t_flow = usl_teacher_dict['flow']
            usl_t_change_256, usl_t_change = usl_teacher_dict['change']
            
            # features for similarity loss, stop gradient to prevent detour flow decoder / half, quater, eight, sixteen
            fs, ft = usl_teacher_dict['feature']
            f_src = [fs[0], fs[1], fs[2], fs[3]]
            f_tgt = [ft[0], ft[1], ft[2], ft[3]]

            # estimated flow and change maps
            u_t_flow1 = usl_t_flow[-1]
            u_t_flow3 = usl_t_flow_256[-1]

            u_t_cng_logit1 = usl_t_change[-1]
            u_t_cng1= u_t_cng_logit1.sigmoid()
            
            u_t_cng_logit3 = usl_t_change_256[-1]
            u_t_cng3= u_t_cng_logit3.sigmoid()
            
            # fused feature similarity loss (shallow feat -> edge, many modes / deep feature -> abstract info., few modes)
            eps = 1e-2             
            f_cosim, vmask = fused_feat_sim(f_src, f_tgt, u_t_flow1, h_original, mode='0-1')
            f_dsim = 1.0 - torch.clamp(alpha*(f_cosim-vmask.float()+1.0) + beta, 0.0, 1.0)
            pseudo_cng = torch.clamp(1.0*f_dsim, min=0.0, max=1.0).detach()
            # f_dsim = 1.0 - (f_cosim-vmask.float()+1.0)
            loss_feat = mask_average(f_dsim, vmask*(1.-u_t_cng1), wavg=True)
            
            # change regularization
            assert u_t_cng1.max() <= 1.0, 'usl max value exceeds 1.0'
            assert u_t_cng1.min() >= 0.0, 'usl min value is below 0.0'
            # loss_cng_reg = -((1.0-pseudo_cng)*(1.0-u_t_cng1+eps).log()).mean()
            loss_cng_reg = -((1.0-pseudo_cng)*F.logsigmoid(-u_t_cng_logit1)).mean()
            # loss_cng_reg += -((pseudo_cng.ge(0.5))*(u_t_cng1+eps).log()).mean()
            
            u_t_flow_orig = F.interpolate(u_t_flow1, size=(h_original, w_original), mode='bilinear')
            usl_src = denormalize(usl_src_image)
            usl_wsrc = warp(usl_src, u_t_flow_orig)
            
            usl_tgt = denormalize(usl_tgt_image)
            u_t_cng_orig = F.interpolate(u_t_cng_logit1, size=(h_original, w_original), mode='bilinear').sigmoid()
            assert u_t_cng_orig.max() <= 1.0, 'usl intp max value exceeds 1.0'
            assert u_t_cng_orig.min() >= 0.0, 'usl intp min value is below 0.0'
            
            if smooth_map == 'edge':
                # _, usl_tgt = edge_detector(usl_tgt)
                usl_tgt = edge_detect(usl_tgt)
                usl_wsrc = edge_detect(usl_wsrc.detach())
                edge_tot = torch.clamp(usl_tgt+usl_wsrc, min=0.0, max=1.0)
                pseudo_cng_orig = F.interpolate(pseudo_cng, size=(520, 520), mode='bilinear')
                # usl_tgt = usl_tgt * pseudo_cng_orig.ge(0.45).float()
 
            # smoothness regularization
            loss_cng_smooth = ms_smooth_reg(img=usl_tgt, outs=[u_t_cng_orig],
                                             img_type=smooth_map, out_type='cng',
                                             weights=[1.], apply_reg='edge', weight_map=pseudo_cng_orig)
             
            loss_total += l_f * loss_feat
            loss_total += l_cr * loss_cng_reg
            loss_total += l_sm * loss_cng_smooth
            
            if args.tr_type == 'semi':
                usl_s_flow_256, usl_s_flow = usl_student_dict['flow']
                usl_s_change_256, usl_s_change = usl_student_dict['change']
                
                loss_semi_flow = multiscaleEPE(usl_s_flow, u_t_flow1.detach(), weights=weights_original, sparse=False,
                                  mean=True, robust_L1_loss=robust_l1_loss,use_flow=None)
                loss_semi_flow += multiscaleEPE(usl_s_flow_256, u_t_flow3.detach(), weights=weights_256, sparse=False,
                                        mean=True, robust_L1_loss=robust_l1_loss,use_flow=None)

                loss_semi_change = multiscaleCE(usl_s_change_256, u_t_cng3.ge(0.5).detach(), weights=weights_256, multi_class=multi_class)
                loss_semi_change += multiscaleCE(usl_s_change, u_t_cng1.ge(0.5).detach(), weights=weights_original, multi_class=multi_class)
                
                loss_total += l_s_semi*(loss_semi_flow + loss_semi_change)
    
                
        loss_total.backward()
        optimizer.step()
        
        # if t_param == 'ema':
        #     update_ema_variables(net.module.student_dec, net.module.teacher_dec, 0.99, n_iter)

        running_total_loss += loss_total.item()
        running_flow_loss += loss_sup_flow.item()
        running_change_loss += loss_sup_change.item()

        if args.tr_type != 'sup':
            running_feat_loss += loss_feat.item()
            running_usl_cng_mean += usl_t_change[-1].sigmoid().mean().item()
            running_fsim_mean += f_cosim[vmask].mean().item()
            running_cng_reg += loss_cng_reg.item()
            running_cng_smooth += loss_cng_smooth.item()
            msg = (f'l_change:{running_change_loss/(i+1):.3f}/{loss_sup_change.item():.3f} | '
                f'l_flow:{running_flow_loss/(i+1):.3f}/{loss_sup_flow.item():.3f} | '
                f'l_feat:{running_feat_loss/(i+1):.3f}/{loss_feat.item():.3f} | '
                f'{running_usl_cng_mean/(i+1):.3f}/{u_t_cng1.max():.3f} | '
                f'{running_fsim_mean/(i+1):.3f} | '
                f'{running_cng_reg/(i+1):.3f} | '
                f'{running_cng_smooth/(i+1):.3f}')
            if args.tr_type == 'semi':
                r_l_semi_flow += loss_semi_flow.item()
                r_l_semi_change += loss_semi_change.item()
                running_u_s_cng_mean += usl_s_change[-1].sigmoid().mean().item()
                msg = (f'cng_sup/semi:{running_change_loss/(i+1):.3f}/{r_l_semi_change/(i+1):.3f} | '
                    f'flow_sup/semi:{running_flow_loss/(i+1):.3f}/{r_l_semi_flow/(i+1):.3f} | '
                    f'l_feat:{running_feat_loss/(i+1):.3f}/{loss_feat.item():.3f} | '
                    f'{running_u_s_cng_mean/(i+1):.3f}/{usl_s_change[-1].sigmoid().max():.3f} | '
                    f'{usl_t_change[-1].sigmoid().min():.3f}/{usl_t_change[-1].sigmoid().max():.3f} | '
                    f'{sup_t_change[-1].sigmoid().min():.3f}/{sup_t_change[-1].sigmoid().max():.3f} | '
                    f'{running_fsim_mean/(i+1):.3f} | '
                    f'{running_cng_reg/(i+1):.3f} | '
                    f'{running_cng_smooth/(i+1):.3f}')
                
        else:
            msg = (f'l_change:{running_change_loss/(i+1):.3f}/{loss_sup_change.item():.3f} | '
                f'l_flow:{running_flow_loss/(i+1):.3f}/{loss_sup_flow.item():.3f} | ')
        pbar.set_description(msg)
        
    running_total_loss /= len(train_loader)
    running_change_loss /= len(train_loader)
    running_flow_loss /= len(train_loader)


    return dict(total=running_total_loss, change=running_change_loss,
                cl=running_cl_loss, flow=running_flow_loss,
                # accuracy = Acc,
                # IoUs=IoUs,
                # mIoU = mIoU,
                # f1=f1
                )


def validate_epoch(args, net,
                   val_loader,
                   device,
                   epoch,
                   save_path,
                   writer,
                   div_flow=1,
                   loss_grid_weights=None,
                   apply_mask=False,
                   sparse=False,
                   robust_L1_loss=False):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        train_writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
        robust_L1_loss: bool on the loss to use
        sparse: bool on sparsity of ground truth flow field
    Output:
        running_total_loss: total validation loss,
        EPE_0, EPE_1, EPE_2, EPE_3: EPEs corresponding to each level of the network (after upsampling
        the estimated flow to original resolution and scaling it properly to compare to ground truth).

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.

    """
    n_iter = epoch*len(val_loader)
    confmeter = ConfusionMeter(k=net.module.num_class,normalized=False)

    net.eval()
    if loss_grid_weights is None:
        loss_grid_weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    running_total_loss = 0
    if not os.path.isdir(save_path): os.mkdir(save_path)

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        EPE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32, device=device)
        CE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32, device=device)

        for i, sup_batch in pbar:
            source_image, target_image, source_image_256, target_image_256 = pre_process_data(
                sup_batch['source_image'],
                sup_batch['target_image'],
                device=device,
                norm = args.img_norm_type)
            source_change, target_change, source_change_256, target_change_256 = \
                pre_process_change(sup_batch['source_change'],
                                   sup_batch['target_change'],
                                   device=device)
            disable_flow = sup_batch['disable_flow'][..., None,None].to(device) # bs,1,1,1
            out_dict = net(source_image, target_image, source_image_256, target_image_256, disable_flow=disable_flow)
            out_flow_256, out_flow_orig = out_dict['flow']
            out_change_256, out_change_orig = out_dict['change']
            ''' Evaluate Flow '''
            # at original size
            flow_gt_original = sup_batch['flow_map'].to(device)
            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            bs, _, h_original, w_original = flow_gt_original.shape
            mask_gt = sup_batch['correspondence_mask'].to(device)
            weights_original = loss_grid_weights[-len(out_flow_orig):]

            # at 256x256 resolution, b, _, 256, 256
            if sparse:
                flow_gt_256 = sparse_max_pool(flow_gt_original, (256, 256))
            else:
                flow_gt_256 = F.interpolate(flow_gt_original, (256, 256),
                                            mode='bilinear', align_corners=False)
            flow_gt_256[:, 0, :, :] *= 256.0 / float(w_original)
            flow_gt_256[:, 1, :, :] *= 256.0 / float(h_original)
            bs, _, h_256, w_256 = flow_gt_256.shape
            weights_256 = loss_grid_weights[:len(out_flow_256)]

            if apply_mask:
                mask = sup_batch['correspondence_mask'].to(device)  # bxhxw, torch.uint8
                Loss = multiscaleEPE(out_flow_orig, flow_gt_original,
                                     weights=weights_original, sparse=sparse,
                                     mean=False, mask=mask, robust_L1_loss=robust_L1_loss)
                if sparse:
                    mask_256 = sparse_max_pool(mask.unsqueeze(1).float(), (256, 256)).squeeze(1).byte()  # bx256x256
                else:
                    mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                             align_corners=False).squeeze(1).byte()  # bx256x256
                Loss += multiscaleEPE(out_flow_256, flow_gt_256, weights=weights_256,
                                      sparse=sparse,
                                      mean=False, mask=mask_256, robust_L1_loss=robust_L1_loss)
            else:
                Loss = multiscaleEPE(out_flow_orig, flow_gt_original,
                                     weights=weights_original, sparse=False,
                                     mean=False, robust_L1_loss=robust_L1_loss)
                Loss += multiscaleEPE(out_flow_256, flow_gt_256, weights=weights_256, sparse=False,
                                      mean=False, robust_L1_loss=robust_L1_loss)

            # calculating the validation EPE
            for index_reso_original in range(len(out_flow_orig)):
                EPE = div_flow * realEPE(out_flow_orig[-(index_reso_original+1)], flow_gt_original, mask_gt, sparse=sparse)
                EPE_array[index_reso_original, i] = EPE

            for index_reso_256 in range(len(out_flow_256)):
                EPE = div_flow * realEPE(out_flow_256[-(index_reso_256+1)], flow_gt_original, mask_gt,
                                        ratio_x=float(w_original) / float(256.0),
                                        ratio_y=float(h_original) / float(256.0),
                                        sparse=sparse)
                EPE_array[(len(out_flow_orig) + index_reso_256), i] = EPE
            # must be both in shape Bx2xHxW

            if i % 1000 == 0:
                vis_img = plot_during_training2(save_path, epoch, i, False,
                                               h_original, w_original, h_256, w_256,
                                               source_image, target_image, source_image_256, target_image_256, div_flow,
                                               flow_gt_original, flow_gt_256, output_net=out_flow_orig[-1],
                                               output_net_256=out_flow_256[-1],
                                               target_change_original=target_change,
                                               target_change_256=target_change_256,
                                               out_change_orig=out_change_orig,
                                               out_change_256=out_change_256,
                                               return_img=True)
                writer.add_image('val_warping_per_iter', vis_img, n_iter)

            # ''' Evaluate Change '''
            out_change_orig = torch.nn.functional.interpolate(out_change_orig[-1].detach(),
                                                              size=(h_original, w_original), mode='bilinear')
            out_change_orig = out_change_orig.permute(0, 2, 3, 1).reshape(-1, out_change_orig.shape[1])
            target_change = target_change.detach().permute(0, 2, 3, 1).reshape(-1, 1)
            confmeter.add(out_change_orig, target_change.squeeze().long())

            running_total_loss += Loss.item()
            pbar.set_description(
                ' val total_loss: %.1f/%.1f' % (running_total_loss / (i + 1),
                                             Loss.item()))
        mean_epe = torch.mean(EPE_array, dim=1)

    conf = torch.FloatTensor(confmeter.value())
    Acc = 100*(conf.diag().sum() / conf.sum()).item()
    recall = conf[1,1]/(conf[1,0]+conf[1,1])
    precision =conf[1,1]/(conf[0,1]+conf[1,1])
    f1 = 100*2*recall*precision/(recall+precision)
    IoUs, mIoU = IoU(conf)
    IoUs, mIoU = 100 * IoUs, 100 * mIoU


    return dict(total=running_total_loss,mEPEs=mean_epe, accuracy = Acc,
                IoUs=IoUs, mIoU = mIoU, f1=f1)



def test_epoch(args, net,
               test_loader,
               device,
               epoch,
               save_path,
               writer,
               div_flow=1,
               plot_interval=10):
    """
    Test epoch script
    Args:
        net: model architecture
        test_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        train_writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
        robust_L1_loss: bool on the loss to use
        sparse: bool on sparsity of ground truth flow field
    Output:
        running_total_loss: total validation loss,
        EPE_0, EPE_1, EPE_2, EPE_3: EPEs corresponding to each level of the network (after upsampling
        the estimated flow to original resolution and scaling it properly to compare to ground truth).

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.

    """
    
    n_iter = epoch*len(test_loader)
    k = 2 if net.module.num_class == 1 else net.module.num_class
    confmeter = ConfusionMeter(k=k,normalized=False)

    net.eval()  # eval 체크 할 것
    
    alpha = args.f_alpha
    beta = args.f_beta
    
    if not os.path.isdir(save_path): os.mkdir(save_path)
    print('Begin Testing {}'.format(save_path))
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, test_batch in pbar:
            source_image, target_image, source_image_256, target_image_256 = pre_process_data(
                test_batch['source_image'],
                test_batch['target_image'],
                device=device,
                norm = args.img_norm_type)
            source_change, target_change, source_change_256, target_change_256 = \
                pre_process_change(test_batch['source_change'],
                                   test_batch['target_change'],
                                   device=device)


            tot_out_dict = net(source_image, target_image, source_image_256, target_image_256, test=True)
            
            out_dict = tot_out_dict['student'] if args.tr_type == 'semi' else tot_out_dict['teacher']
            out_flow_256, out_flow_orig = out_dict['flow']
            out_change_256, out_change_orig = out_dict['change']

            bs, _, h_original, w_original = source_image.shape
            bs, _, h_256, w_256 = source_image_256.shape
            flow_gt_original = F.interpolate(out_flow_orig[-1], (h_original, w_original),
                                              mode='bilinear', align_corners=False)  # shape Bx2xHxW
            flow_gt_256 = F.interpolate(out_flow_256[-1], (h_256, w_256),
                                              mode='bilinear', align_corners=False)  # shape Bx2xHxW
            
            fs, ft = out_dict['feature']
            f_src = [fs[0], fs[1], fs[2], fs[3]]
            f_tgt = [ft[0], ft[1], ft[2], ft[3]]
            # f_src = [fs[4]]
            # f_tgt = [ft[4]]
            # f_diff, _ = fused_feat_diff(f_src, f_tgt, out_flow_orig[-1], h_original)
            f_cosim, vmask = fused_feat_sim(f_src, f_tgt, out_flow_orig[-1], h_original, mode='0-1')
            eps=1e-5
            pseudo_cng = 1.0 - torch.clip(alpha*(f_cosim-vmask.float()+1.0) + beta, 0.0, 1.0).detach()
            pseudo_cng = torch.clamp(1.0*pseudo_cng, min=0.0, max=1.0)
            pseudo_cng_bi = F.interpolate(pseudo_cng, size=(520, 520), mode='bilinear')
            pseudo_cng_mask = pseudo_cng_bi.ge(0.45).float()
            
            flow_orig = F.interpolate(out_flow_orig[-1], size=(h_original, w_original), mode='bilinear')
            cng_orig = F.interpolate(out_change_orig[-1], size=(h_original, w_original), mode='bilinear').sigmoid()
            usl_wsrc = warp(denormalize(source_image), flow_orig)
            
            edge_src = edge_detect(usl_wsrc)
            edge_tgt = edge_detect(denormalize(target_image)) 
            edge_tot = torch.clamp(edge_src+edge_tgt, max=1.0)

            if i % plot_interval == 0:
                plot_during_training2(save_path, epoch, i, False,
                                        h_original, w_original, h_256, w_256,
                                        source_image, target_image, source_image_256, target_image_256, div_flow,
                                        flow_gt_original, flow_gt_256, output_net=out_flow_orig[-1],
                                        output_net_256=out_flow_256[-1],
                                        target_change_original=target_change,
                                        target_change_256=target_change_256,
                                        out_change_orig=out_change_orig,
                                        out_change_256=out_change_256,
                                        f_sim_map = pseudo_cng_mask, vmask=vmask,
                                        return_img=False,
                                        multi_class=net.module.num_class,
                                        tr_type=args.tr_type,
                                        src_edge=edge_tot*pseudo_cng_mask, tgt_edge=(edge_tgt)*pseudo_cng_mask,
                                        make_vid=args.make_vid)
                
            out_change_orig = torch.nn.functional.interpolate(out_change_orig[-1].detach(),
                                                              size=(h_original, w_original), mode='bilinear')
            vmask = torch.nn.functional.interpolate(vmask.float(), size=(h_original, w_original), mode='nearest')
            
            if net.module.num_class == 1:
                out_change = (torch.sigmoid(out_change_orig)).ge(0.5).long() * vmask.long()
                out_change = out_change.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
            else:
                out_change = out_change_orig.permute(0, 2, 3, 1).reshape(-1, out_change_orig.shape[1])
            
            target_change = target_change.long() * vmask.long()
            target_change = target_change.detach().permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
            
            try:
                confmeter.add(out_change, target_change)
            except:
                print(out_change_orig.shape)
                print(target_change.shape)
                print(net.module.num_class)
                print(out_change_orig.max())

    conf = torch.FloatTensor(confmeter.value())
    Acc = 100*(conf.diag().sum() / conf.sum()).item()
    recall = conf[1,1]/(conf[1,0]+conf[1,1])
    precision =conf[1,1]/(conf[0,1]+conf[1,1])
    f1 = 100*2*recall*precision/(recall+precision)
    IoUs, mIoU = IoU(conf)
    IoUs, mIoU = 100 * IoUs, 100 * mIoU

    return dict(accuracy = Acc, IoUs=IoUs, mIoU = mIoU, f1=f1, precision=precision, recall=recall)
