#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
# import sys
# print('sys.path = ', sys.path)
# sys.path.append('/data1/hn/gaussianSim/gs4d/gs_1/submodules/depth-diff-gaussian-rasterization')
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine",return_decomposition=False,return_dx=False,render_feat=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0  # 不是很明白？？
    try:
        screenspace_points.retain_grad() # 明白了，训练的时候需要计算梯度，eval()的时候不需要
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz # ???

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) 
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5) #计算相机的视场角
    # 下面这块代码仿照的pytorch3d的代码，设置相机渲染的参数
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height), 
        image_width=int(viewpoint_camera.image_width), # 根据之前的load_size设置的
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color, # 透明情况下为[0,0,0]
        scale_modifier=scaling_modifier, #???反正目前是1.0
        viewmatrix=viewpoint_camera.world_view_transform.cuda(), #???
        projmatrix=viewpoint_camera.full_proj_transform.cuda(), #??? 猜测是3D投相机u,v的完整矩阵
        sh_degree=pc.active_sh_degree, #这个是当前激活的SH的阶数，起始的时候是0
        campos=viewpoint_camera.camera_center.cuda(), #相机的中心位置
        prefiltered=False, #??? 这个是啥
        debug=pipe.debug #??? 开关有什么区别
    ) # 设置rasterizer的参数（主要都是相机相关的）
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings) # 根据着色器设置rasterizer

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation


    means2D = screenspace_points # 这边好多都和我之前写可微渲染的时候的操作很像,点云数量*3的tensor
    opacity = pc._opacity # 这个是点云的透明度，点云数量*1的tensor，好像是0.1逆sigmoid得到的
    shs = pc.get_features # 这个是SH的系数，点云数量*(l+1)^2*3的tensor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python: # ??? 暂时为False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:   
        scales = pc._scaling # 点云数量*3的tensor,高斯球的缩放
        rotations = pc._rotation # 点云数量*4的tensor,高斯球的旋转，四元数，目前全为0
    deformation_point = pc._deformation_table # 因为没有分割，所以全为True
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs # 对于corse阶段，直接使用点云初始化的参数
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, dx, feat, dshs = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final) # 用exp归一化
    rotations_final = pc.rotation_activation(rotations_final) # 归一化以防出现不规则的四元数
    opacity = pc.opacity_activation(opacity_final) # 用sigmoid归一化
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None: # 是None
        if pipe.convert_SHs_python:
            shs_view = shs_final.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) # 其实就是转成num_points*3*(l+1)^2的tensor
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1)) #?? 这个是点云的xyz坐标减去相机的xyz坐标,但是repeat的维度是点云的数量，等于repeat了个寂寞
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # 求出入射方向
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)  #？？？这里不转换成相机坐标系的入射方向吗
            # print(sh2rgb.max())
            # print(sh2rgb.min())
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) # 还是不明白，这个干嘛用的
            # print(colors_precomp.max())
            # print(colors_precomp.min())
        else:
            pass
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    if colors_precomp is not None:
        shs_final = None
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final, #高斯椭球的中心点坐标，也即3D Gaussian的均值
        means2D = means2D, # 3D Gaussian的均值在屏幕上的投影坐标，但是好像是全0的
        shs = shs_final,  # SH系数，但是由于之前算了color，所以这里是None
        colors_precomp = colors_precomp, # [N,3] 根据之前的sh系数算出来的
        opacities = opacity, # [N,1] 高斯椭球的透明度，全为0.1
        scales = scales_final, # [N,3] 高斯椭球的缩放系数
        rotations = rotations_final, # [N,4] 高斯椭球的旋转系数,全为标准四元数
        cov3D_precomp = cov3D_precomp) # ??暂时为None
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    result_dict = {}
    
    result_dict.update({
        "render": rendered_image, # [H,W,3] 这个是渲染出来的图像
        "viewspace_points": screenspace_points, # 不是很明白，全为0
        "visibility_filter" : radii > 0, # [N] 这个是可见性过滤器，radii(投影到2D的覆盖范围)大于0的点
        "radii": radii, # [N] 这个是高斯椭球在屏幕上的半径
        "depth":depth}) # 深度图

    features_precomp = None
    # Concatenate the pre-computation colors and CLIP features indices
    # render_feat = True
    if render_feat and "fine" in stage:
        colors_precomp = feat
        shs_final = None
        rendered_image2, _, _ = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp, # [N,3]
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)
        
        result_dict.update({"feat": rendered_image2})

    if return_decomposition and dx is not None: # 暂时为None
        dx_abs = torch.abs(dx) # [N,3]
        max_values = torch.max(dx_abs, dim=1)[0] # [N]
        thre = torch.mean(max_values)
        
        dynamic_mask = max_values > thre
        # dynamic_points = np.sum(dynamic_mask).item()
        
        rendered_image_d, radii_d, depth_d = rasterizer(
            means3D = means3D_final[dynamic_mask],
            means2D = means2D[dynamic_mask],
            shs = shs_final[dynamic_mask] if shs_final is not None else None,
            colors_precomp = colors_precomp[dynamic_mask] if colors_precomp is not None else None, # [N,3]
            opacities = opacity[dynamic_mask],
            scales = scales_final[dynamic_mask],
            rotations = rotations_final[dynamic_mask],
            cov3D_precomp = cov3D_precomp[dynamic_mask] if cov3D_precomp is not None else None)
        
        rendered_image_s, radii_s, depth_s = rasterizer(
            means3D = means3D_final[~dynamic_mask],
            means2D = means2D[~dynamic_mask],
            shs = shs_final[~dynamic_mask] if shs_final is not None else None,
            colors_precomp = colors_precomp[~dynamic_mask] if colors_precomp is not None else None, # [N,3]
            opacities = opacity[~dynamic_mask],
            scales = scales_final[~dynamic_mask],
            rotations = rotations_final[~dynamic_mask],
            cov3D_precomp = cov3D_precomp[~dynamic_mask] if cov3D_precomp is not None else None
            )
        
        result_dict.update({
            "render_d": rendered_image_d,
            "depth_d":depth_d,
            "visibility_filter_d" : radii_d > 0,
            "render_s": rendered_image_s,
            "depth_s":depth_s,
            "visibility_filter_s" : radii_s > 0,
            })
        
    if return_dx and "fine" in stage:
        result_dict.update({"dx": dx})
        result_dict.update({'dshs' : dshs})

    return result_dict