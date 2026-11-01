# ------------------------------------------------------------------------
# O1O: Grouping Known Classes to Identify Unknown Objects as Odd-One-Out
# Misra Yavuz, Fatma Guney. ACCV, 2024. (https://arxiv.org/abs/2410.07514)
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection 
# Orr Zohar, Jackson Wang, Serena Yeung
# -----------------------------------------------------------------------
# Modified from OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import sys
sys.path.insert(0, "/dev/shm")  # 强制优先加载 /dev/shm 下含 route 分支的 fotg_module/fotg_losses，避免被 O1O-main 同名旧文件(cwd 优先)遮蔽

# 用 importlib 从 /dev/shm 绝对路径硬加载 fotg_losses/fotg_module 并注册到 sys.modules，
# 彻底绕开 sys.path 后续被 O1O-main(cwd) 插回最前导致的旧版遮蔽（route 分支必须来自 /dev/shm 版）
import importlib.util as _ilu
import os as _os

def _load_fotg_abs(name, rel):
    _p = _os.path.join("/dev/shm", rel)
    _spec = _ilu.spec_from_file_location(name, _p)
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)  # exec_module 在 spec.loader 上，不在 importlib.util 上
    sys.modules[name] = _m
    return _m

_load_fotg_abs("fotg_losses", "fotg_losses.py")
_load_fotg_abs("fotg_module", "fotg_module.py")
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world_o1o import OWDetection
from engine import evaluate, train_one_epoch, get_exemplar_replay, create_ft_dataset
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    ################ Deformable DETR ################
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--lr_drop', default=35, type=int)
    # parser.add_argument('--lr_drop_epochs', default=51, type=int) 
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--masks', default=False, action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone', default='dino_resnet50', type=str, help="Name of the convolutional backbone to use")
    # Model parameters
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # Transformer
    parser.add_argument('--return_interm_layers', action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str, help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--num_results', default=300, type=int, help="Number of detection results")
    parser.add_argument('--pre_norm', action='store_true',  help="Using pre-norm in the Transformer blocks.")    
    parser.add_argument('--num_select', default=300, type=int,  help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int, help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', help="Random init the x,y of anchor boxes and freeze them.")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    # Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")
    # Training
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    
    ################ O1O ################
    # dataset
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int)
    parser.add_argument('--invalid_cls_logits', default=False, action='store_true', help='owod setting')
    parser.add_argument('--train_set', default='', help='training txt files')
    parser.add_argument('--test_set', default='', help='testing txt files')
    parser.add_argument('--num_classes', default=81, type=int)
    parser.add_argument('--dataset', default='OWDETR', help='defines which dataset is used. Built for: {TOWOD, OWDETR, VOC2007}')
    parser.add_argument('--data_root', default='./data/OWOD', type=str)

    # superclasses
    parser.add_argument('--num_superclass', default=1, type=int)
    parser.add_argument('--prev_num_superclass', default=1, type=int)
    parser.add_argument('--cur_num_superclass', default=1, type=int)
    parser.add_argument('--invalid_sup_logits', default='', type=str)

    # pseudo-labels
    parser.add_argument('--pseudo_path', default='', type=str)
    parser.add_argument('--alternative_pseudo_path', default='', type=str)
    parser.add_argument('--pseudo_threshold', default=0.0, type=float)
    parser.add_argument('--pseudo_nms_iou', default=0.7, type=float)
    parser.add_argument('--pseudo_backup', default='', type=str)
    parser.add_argument('--alternative_pseudo_backup', default='', type=str)
    parser.add_argument('--num_unk_objects', default=20, type=int, help="")
    parser.add_argument('--max_num_objects', default=64, type=int, help="")

    # dn dab args
    parser.add_argument('--use_dn', action="store_true", help="use denoising training.")
    parser.add_argument('--scalar', default=5, type=int, help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.2, type=float, help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float, help="box noise scale to shift and scale")
    parser.add_argument('--contrastive', action="store_true", help="use contrastive training.")
    parser.add_argument('--use_mqs', action="store_true", help="use mixed query selection from DINO.")
    parser.add_argument('--use_lft', action="store_true", help="use look forward twice from DINO.")
    parser.add_argument('--amp', action='store_true', help="Train with mixed precision")
    parser.add_argument('--save_results', action='store_true', help="For eval only. Save the outputs for all images.")
    parser.add_argument('--debug', action='store_true', help="For debug only. It will perform only a few steps during trainig and val.")

    # training
    parser.add_argument('--model_type', default='prob_dab_deformable_detr', type=str)    
    parser.add_argument('--momentum', default=0.1, type=float)
    parser.add_argument('--empty_weight', default=0.1, type=float)
    parser.add_argument('--lr_drop_gamma', default=0.1, type=float)
    parser.add_argument('--num_gpus', default=0, type=int)
    parser.add_argument('--sup_loss_coef', default=1, type=float)
    parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
    
    # exemplar replay
    parser.add_argument('--num_inst_per_class', default=50, type=int, help="number of instances per class")
    parser.add_argument('--exemplar_replay_selection', default=False, action='store_true', help='use learned exemplar selection')
    parser.add_argument('--exemplar_replay_max_length', default=1e10, type=int, help="max number of images that can be saves")
    parser.add_argument('--exemplar_replay_dir', default='', type=str, help="directory of exemplar replay txt files")
    parser.add_argument('--exemplar_replay_prev_file', default='', type=str, help="path to previous ft file")
    parser.add_argument('--exemplar_replay_cur_file', default='', type=str, help="path to current ft file")
    parser.add_argument('--exemplar_replay_random', default=False, action='store_true', help='make selection random')
    
    # eval and logging
    parser.add_argument('--eval_threshold', default=0.0, type=float)
    parser.add_argument('--wandb_name', default='', type=str)
    parser.add_argument('--wandb_project', default='O1O', type=str)

    # ---- Fast Episodic Query Memory + Generalization Gate + Selective Query Plasticity ----
    # Every switch defaults OFF so the model is exactly the original O1O.
    # Experimental modes (see task spec):
    #   baseline      : all OFF
    #   adapter-only  : --use_plastic_adapter            (gate g = 1)
    #   memory-only   : --use_fast_memory
    #   full          : --use_fast_memory --use_plastic_adapter  (g = u * sigmoid((r-tau)/T))
    parser.add_argument('--task_id', default=1, type=int, help='current OWOD task id (for rep logging)')
    parser.add_argument('--use_fast_memory', default=False, action='store_true',
                        help='Enable Fast Episodic Query Memory (query-level, independent of image-level exemplar replay).')
    parser.add_argument('--use_plastic_adapter', default=False, action='store_true',
                        help='Enable Selective Query Adapter (init-0 residual, learned via original O1O losses).')
    parser.add_argument('--fp_gate_type', default='unknown_x_memory_consistency', type=str,
                        choices=['unknown_x_memory_consistency', 'memory_consistency', 'unknown_only', 'stable_reference', 'current_space_novelty'],
                        help='Generalization gate formulation. "current_space_novelty" = V3 '
                             '(alpha = n_t * c_t, n_t from current-space old prototypes).')
    parser.add_argument('--fp_memory_size', default=256, type=int, help='Max Fast Memory slots.')
    parser.add_argument('--fp_memory_momentum', default=0.1, type=float, help='EMA momentum for memory update.')
    parser.add_argument('--fp_memory_tau_create', default=0.7, type=float, help='Create new slot if max cosine < tau.')
    parser.add_argument('--fp_gate_tau', default=0.7, type=float, help='Gate temperature tau.')
    parser.add_argument('--fp_gate_temperature', default=0.1, type=float, help='Gate sigmoid temperature.')
    parser.add_argument('--fp_adapter_bottleneck', default=32, type=int, help='Adapter bottleneck dim.')
    parser.add_argument('--fp_adapter_scale', default=1.0, type=float, help='Adapter residual scale.')
    parser.add_argument('--fp_rep_log', default=False, action='store_true',
                        help='Save per-positive-query (q, q_plastic, label) for representation analysis.')
    parser.add_argument('--fp_rep_log_dir', default='', type=str, help='Directory for representation buffers (use /dev/shm).')
    parser.add_argument('--fp_reference_rep', default='', type=str,
                        help='Path to a representation buffer (.pt) used to build the frozen old-class '
                             'reference for gate_type=stable_reference. Queries with class_label < '
                             'PREV_INTRODUCED_CLS are used as the stable reference.')
    parser.add_argument('--fp_memory_load', default='', type=str, help='Path to load Fast Memory state before training.')
    parser.add_argument('--fp_memory_save', default='', type=str, help='Path to save Fast Memory state after training.')
    parser.add_argument('--fp_debug', default=False, action='store_true',
                        help='Compute & log extended mechanistic diagnostics: fraction_gate>0.1/>0.5, '
                             'max adapter residual, memory hit rate, mean r on hits, and '
                             'novel-first vs novel-repeated vs old-known gate (full probe only).')
    parser.add_argument('--fp_fast_lr_mult', default=5, type=float,
                        help='V3 fast/slow timescale: learning-rate multiplier for the fast adapter '
                             '(fp.adapter.*) vs the slow O1O params. lr_fast = args.lr * fp_fast_lr_mult. '
                             'Test 1x / 5x / 10x (default 5).')
    parser.add_argument('--fp_consistency_kappa', default=5.0, type=float,
                        help='V3 evidence-accumulation temperature kappa: c_t = 1 - exp(-N_m / kappa), '
                             'where N_m is the pre-update fast-memory slot hit count.')

    # ---- Fast-Slow dual pathway + SCM (CLS-inspired) ----
    parser.add_argument('--use_fast_slow', action='store_true',
                        help='Enable CLS-inspired Fast-Slow dual visual encoder + SCM on top of O1O.')
    parser.add_argument('--fast_slow_bottleneck', default=64, type=int,
                        help='Bottleneck dim of the Fast/Slow bottleneck adapters.')
    parser.add_argument('--fast_slow_heads', default=8, type=int,
                        help='Number of heads in the SCM cross-attention.')
    parser.add_argument('--scm_share_base', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='SCM fusion uses delta(F_fast-F_slow) when True; direct F_fast when False.')
    parser.add_argument('--fast_lr_mult', default=5.0, type=float,
                        help='Fast adapter lr = args.lr * fast_lr_mult (plasticity timescale).')
    parser.add_argument('--ema_rho', default=0.01, type=float,
                        help='EMA consolidation rate: slow adapter <- (1-rho)*slow + rho*fast per step.')
    parser.add_argument('--scm_loss_w', default=0.0, type=float,
                        help='Weight for SCM consolidation auxiliary loss (0 = off).')

    # ---- Bio-inspired Semi-Orthogonal Incremental Learning (component A) ----
    parser.add_argument('--sep_loss_w', default=0.0, type=float,
                        help='Weight for semi-orthogonal separation loss on class_embed[-1] '
                             'rows: penalizes overlap between old-class ([:PREV]) and '
                             'current-class ([PREV:PREV+CUR]) classification directions. '
                             '0 = off (original O1O behaviour).')
    parser.add_argument('--fotg_on', default=0, type=int,
                        help='FOTG 约束开关。0 = 完全不执行（默认，不改原行为）。')
    parser.add_argument('--fotg_old_ckpt', default='', type=str,
                        help='冻结旧模型(old_model)的权重来源。'
                             '留空 = 沿用旧行为（deepcopy 当前模型，此时参照系是"本次训练起点"）；'
                             '显式给出上一个任务的 ckpt 路径时，old_model 由该 ckpt 加载，'
                             'F 约束才是"保持上一任务旧类功能"的正确语义。'
                             '续跑(resume 非首任务 ckpt)时必须显式指定，否则参照系会被重置为续跑起点。')
    parser.add_argument('--fotg_dim', default=256, type=int, help='query 特征维度 hidden_dim。')
    parser.add_argument('--fotg_k_sub', default=8, type=int, help='每类局部切空间 PCA 维度 k。')
    parser.add_argument('--fotg_w_threshold', default=0.0, type=float,
                        help='旧类 query 置信度阈值。默认 0 = 不设绝对阈值（只用每图 top-k）。')
    parser.add_argument('--fotg_w_topk', default=10, type=int, help='每张图保留的旧类 query 数。')
    parser.add_argument('--fotg_w_soft', default=1, type=int, help='1=用置信度软加权，0=0/1 掩码。')
    parser.add_argument('--fotg_eta_box', default=1.0, type=float, help='模块 F 中定位项权重。')
    parser.add_argument('--fotg_O_0', default=0.5, type=float, help='模块 O 基准重叠（建议用实测标定值）。')
    parser.add_argument('--fotg_alpha_sup', default=0.1, type=float, help='同 superclass 的增量。')
    parser.add_argument('--fotg_beta_conf', default=0.1, type=float, help='跨 superclass 冲突的减量。')
    parser.add_argument('--fotg_O_min', default=0.2, type=float)
    parser.add_argument('--fotg_O_max', default=0.8, type=float)
    parser.add_argument('--fotg_eta_pd1', default=1.0, type=float, help='模块 T 中 H1 项权重。')
    parser.add_argument('--fotg_min_persistence', default=0.0, type=float)
    parser.add_argument('--fotg_topo_every', default=50, type=int, help='每多少步算一次拓扑约束。')
    parser.add_argument('--fotg_eps_fib', default=1e-3, type=float, help='C_fib 的预算 ε。')
    parser.add_argument('--fotg_eps_ov', default=1e-2, type=float, help='C_ov 的预算 ε。')
    parser.add_argument('--fotg_eps_topo', default=0.5, type=float, help='C_topo 的预算 ε。')
    parser.add_argument('--fotg_lam_fib', default=0.0, type=float, help='λ_fib 初值（0=纯对偶上升从 0 起）。')
    parser.add_argument('--fotg_lam_ov', default=0.0, type=float)
    parser.add_argument('--fotg_lam_topo', default=0.0, type=float)
    parser.add_argument('--fotg_lam_route', default=0.0, type=float, help='λ_route 初值（Phase1 路由感知功能保持；0=关闭）。')
    parser.add_argument('--fotg_eps_route', default=0.05, type=float, help='C_route 的预算 ε（路由分数漂移）。')
    parser.add_argument('--fotg_lam_unit_route', default=1.0, type=float)
    parser.add_argument('--fotg_dual_lr_route', default=1.0, type=float)
    parser.add_argument('--fotg_lam_max_route', default=1e3, type=float)
    parser.add_argument('--fotg_unk_w', default=0.0, type=float, help='模块 U 的 BCE 权重。')
    parser.add_argument('--fab_on', default=0, type=int,
                        help='F-B Jacobian 软投影开关。0=关闭（默认）。1=在 backward 阶段把'
                             '特征空间梯度投影到旧读出雅可比零空间，压低易改变旧任务输出的更新方向。')
    parser.add_argument('--fab_lambda', default=1e-3, type=float,
                        help='F-B 投影正则 λ：P = I - Jᵀ(JJᵀ+λI)⁻¹J。λ 越大投影越软（保留越多梯度）。')
    parser.add_argument('--fab_soft', default=1.0, type=float,
                        help='F-B soft-projection coefficient alpha: P = I - alpha*J^T(JJ^T+lam*I)^{-1}J. 1=hard orthogonal; 0.5=half-strength soft; ->0 off.')
    parser.add_argument('--fotg_dual', default=1, type=int, help='1=启用对偶上升更新 λ。')
    parser.add_argument('--fotg_dual_every', default=20, type=int)
    parser.add_argument('--fotg_accum_every', default=20, type=int, help='几何记忆累积间隔。')
    parser.add_argument('--margin_on', default=0, type=int,
                        help='Old-New margin 保持损失开关。0=关闭。1=在旧类 query 上约束'
                             ' z_old - max_{c∈新类} z_c 不低于教师(T1)参照间隔，保护跨类判别边界。')
    parser.add_argument('--margin_lam', default=0.1, type=float,
                        help='L_margin 权重。L = λ·mean max(0, m_i^{T1} - m_i^{T2})。')
    parser.add_argument('--margin_eps', default=0.0, type=float,
                        help='margin 松弛 ε：允许当前 margin 比 T1 低 ε 再开始惩罚（默认 0=严格保持）。')

    # ---- 新 F-B：PCA-directional Jacobian Protection（参数空间软投影） ----
    parser.add_argument('--pca_fab_on', default=0, type=int,
                        help='新 F-B 开关：对生成 hs_last 的可训练参数 θ（排除读头）的梯度，'
                             '沿预计算 PCA 冲突方向做软投影 g\' = g - J_K^T(J_KJ_K^T+λI)^{-1}J_K g。'
                             '0=关闭（默认），1=开启。与旧 --fab_on 独立。')
    parser.add_argument('--pca_basis_pt', default='', type=str,
                        help='预计算 PCA 方向基 .pt 路径（含 mu [D]、V [D,64]、可选 dirs）。')
    parser.add_argument('--pca_dirs', default='', type=str,
                        help='冲突方向索引，逗号分隔，如 "5,9,37"。空=用 .pt 里的 dirs。')
    parser.add_argument('--pca_fab_lam', default=1e-3, type=float,
                        help='新 F-B 投影正则 λ（J_K J_K^T + λI 的对角项）。')
    parser.add_argument('--pca_fab_every', default=20, type=int,
                        help='新 F-B 每 N 步重算一次 J_K 并投影，中间步复用缓存 J_K（省算力）。')

    # ---- MFFC-F：多通道功能纤维约束（新版 F 模块，--mffc_f_on 1 时启用） ----
    parser.add_argument('--mffc_f_on', default=0, type=int,
                        help='MFFC-F（Multi-channel Functional Fibre Constraint）开关。'
                             '0=关闭（默认）；1=在 FOTG 块内计算 F1 身份/F2 前景/F3 定位 三通道'
                             '功能保持损失 + 可选 L_fib 局部 Jacobian 纤维约束。'
                             '需配合 --fotg_on 1（或本开关一并触发）以加载冻结旧模型 teacher。')
    parser.add_argument('--mffc_f_topk', default=5, type=int,
                        help='F1 使用的教师 Top-K 旧类竞争类数。')
    parser.add_argument('--mffc_f_iou', default=0.5, type=float,
                        help='GT 对齐匹配的 IoU 阈值（student/teacher 各自独立匹配同一 GT）。')
    parser.add_argument('--mffc_f_lam_id', default=1.0, type=float, help='F1 身份保持权重。')
    parser.add_argument('--mffc_f_lam_fg', default=1.0, type=float, help='F2 前景保持权重。')
    parser.add_argument('--mffc_f_lam_loc', default=1.0, type=float, help='F3 定位保持权重。')
    parser.add_argument('--mffc_f_lam_fib', default=0.1, type=float,
                        help='L_fib 局部 Jacobian 功能纤维约束权重（0=关闭该功能正则）。')
    parser.add_argument('--mffc_f_w', default=1.0, type=float,
                        help='MFFC-F 总损失整体权重（乘到 L_id+L_fg+L_loc+L_fib 上）。')
    parser.add_argument('--mffc_f_loc_scale', default=1.0, type=float,
                        help='定位坐标额外缩放（归一化坐标用 1.0；像素坐标需调大）。')
    parser.add_argument('--mffc_f_fib_w_fg', default=1.0, type=float,
                        help='L_fib 中前景通道独立归一化权重 W_F（满足三通道独立归一化边界）。')
    parser.add_argument('--mffc_f_fib_w_loc', default=1.0, type=float,
                        help='L_fib 中定位通道独立归一化权重 W_F。')
    parser.add_argument('--mffc_f_temp', default=1.0, type=float,
                        help='身份 margin 的温度尺度。')

    parser.add_argument('--prealloc_gib', default=0.0, type=float,
                        help='Pre-allocate N GiB of CUDA memory BEFORE build_model '
                             '(held only during model/optimizer construction to avoid the '
                             'memory being grabbed by other processes), then released. 0 = off.')

    return parser

def main(args):

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if len(args.wandb_project)>0 and utils.is_main_process():
        import wandb
        if len(args.wandb_name)>0:
            wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
        else:
            wandb.init(project=args.wandb_project,config=args)
        #wandb.config = args
    else:
        wandb=None

    print(args)

    device = torch.device(args.device)
    if args.num_gpus == 1:
        torch.cuda.set_device(device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 预分配显存占位（仅在 build_model 期间持有，防止被其他进程抢走）
    _prealloc_holder = None
    if getattr(args, 'prealloc_gib', 0.0) > 0 and 'cuda' in str(device):
        _n = int(args.prealloc_gib * 1024**3 / 4)
        _prealloc_holder = torch.empty(_n, device=device, dtype=torch.float32)
        torch.cuda.synchronize(device)
        _f, _t = torch.cuda.mem_get_info(device)
        print('[prealloc] held %.1f GiB, free=%.1f/%.1f GiB' %
              (args.prealloc_gib, _f/1024**3, _t/1024**3), flush=True)

    model, criterion, postprocessors, exemplar_selection = build_model(args, mode=args.model_type)
    model.to(device)

    # 模型与优化器已构建完成，释放占位，交还给训练主循环
    if _prealloc_holder is not None:
        del _prealloc_holder
        torch.cuda.empty_cache()
        print('[prealloc] holder released after build_model', flush=True)

    # Fast-Slow + SCM: scm_loss_w==0 时 consolidation_loss 不计入 loss，cross-attn 输出 f_ca
    # 无任何消费者；关闭其前向以省下高分辨率 C3 上 [B, heads, N, N] 注意力矩阵显存（数 GiB，OOM 元凶）。
    if getattr(args, 'use_fast_slow', False) and getattr(model, 'scm', None) is not None:
        _use_ca = getattr(args, 'scm_loss_w', 0.0) > 0
        for _m in model.scm:
            _m.use_cross_attn = _use_ca
        print(f'[FastSlow] SCM cross-attn enabled={_use_ca} (scm_loss_w={getattr(args, "scm_loss_w", 0.0)})')

    # Fast Plasticity: tag task id and (optionally) load a previously saved
    # Fast Memory state. Loading is independent of the O1O checkpoint format.
    if getattr(model, 'fp', None) is not None:
        model.current_task_id = args.task_id
        if args.fp_memory_load:
            model.load_fp_memory(args.fp_memory_load)
            print('[FastPlasticity] loaded memory from', args.fp_memory_load)
        if getattr(args, 'fp_reference_rep', ''):
            model.fp.load_reference(args.fp_reference_rep, prev_introduced=args.PREV_INTRODUCED_CLS)
            print('[FastPlasticity] loaded stable reference from', args.fp_reference_rep)

    model_without_ddp = model
    print(model_without_ddp)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train, dataset_val = get_datasets(args) 
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, 
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, 
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, 
                                 args.batch_size, 
                                 sampler=sampler_val,
                                 drop_last=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if (not match_name_keywords(n, args.lr_backbone_names)
                     and not match_name_keywords(n, args.lr_linear_proj_names)
                     and 'fp.adapter.' not in n
                     and 'fast_path.adapters' not in n
                     and 'scm.' not in n
                     and p.requires_grad)],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]
    # Fast Plasticity: the Selective Query Adapter (fp.adapter.*) is the FAST
    # timescale. It gets its own optimizer group with lr_fast = lr * fp_fast_lr_mult
    # (e.g. slow 2e-5, fast 1e-4) so it tracks new classes faster than the slowly
    # consolidating O1O backbone -- realising the fast/slow division without
    # touching weight decay or any other O1O hyperparameter.
    adapter_params = [p for n, p in model_without_ddp.named_parameters()
                      if 'fp.adapter.' in n and p.requires_grad]
    if adapter_params:
        param_dicts.append({
            "params": adapter_params,
            "lr": args.lr * args.fp_fast_lr_mult,
        })
    # Fast-Slow + SCM: the Fast adapter (fast_path.adapters.*) is the plastic
    # timescale (lr_fast = lr * fast_lr_mult); SCM params use the normal lr.
    fast_adapter_params = [p for n, p in model_without_ddp.named_parameters()
                           if 'fast_path.adapters' in n and p.requires_grad]
    if fast_adapter_params:
        param_dicts.append({
            "params": fast_adapter_params,
            "lr": args.lr * args.fast_lr_mult,
        })
    scm_params = [p for n, p in model_without_ddp.named_parameters()
                  if 'scm.' in n and p.requires_grad]
    if scm_params:
        param_dicts.append({
            "params": scm_params,
            "lr": args.lr,
        })
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.lr_drop_gamma)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    output_dir = Path(args.output_dir)

    if args.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load( args.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(msg)
        # Fast-Slow: align Fast adapter to Slow (zero-residual start) after loading.
        if getattr(model_without_ddp, 'use_fast_slow', False):
            model_without_ddp.sync_fast_from_slow()

        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, dataset_val, device, args.output_dir, args=args,
                                            eval_mode=True, eval_threshold=args.eval_threshold)
            if wandb is not None and utils.is_main_process():
                wandb.log({str(key): val for key, val in test_stats["metrics"].items()})

        # Fast Plasticity: flush the representation buffer accumulated during an
        # --eval pass. The per-epoch save lives only in the training loop, so an
        # eval-only run would otherwise discard every rep sample. Only acts when
        # rep logging is enabled. Memory is NOT mutated here (training=False), so
        # the logged memory_consistency stays consistent across the eval pass.
        if getattr(model_without_ddp, 'fp', None) is not None and args.fp_rep_log and args.fp_rep_log_dir:
            rep_path = os.path.join(args.fp_rep_log_dir,
                                    f'fp_rep_task{args.task_id}_eval.pt')
            if model_without_ddp.save_fp_representation(rep_path):
                print('[FastPlasticity] saved eval representation ->', rep_path)
            model_without_ddp.dump_fp_gate_by_class(
                os.path.join(args.fp_rep_log_dir,
                             f'fp_gate_by_class_task{args.task_id}_eval.json'))

        if args.exemplar_replay_selection:
            with torch.no_grad():
                model.eval()        
                image_sorted_scores = get_exemplar_replay(model, exemplar_selection, device, data_loader_train, args)
                create_ft_dataset(args, image_sorted_scores)        
        return
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        # Fast-Slow / FP new params are expected to be absent from the checkpoint
        # (they are added on top of O1O and init-0 / random). Filter them out so the
        # missing-key report only shows genuine problems.
        _fs_keys = ('fast_path.', 'slow_path.', 'scm.')
        missing_keys = [k for k in missing_keys if not k.startswith(_fs_keys)]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        # When any FastPlasticity / Fast-Slow module is active, the checkpoint has no
        # optimizer state for the new params; loading it would crash (param-group
        # mismatch) and could drop the new params from the optimizer. Start a fresh
        # optimizer instead so all modes share an identical optimizer starting point.
        _fp_active = getattr(model_without_ddp, 'fp', None) is not None
        _skip_opt = _fp_active or getattr(model_without_ddp, 'use_fast_slow', False)
        if not args.eval and (not _skip_opt) and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)

            # ---- plain-O1O resume fix: previously start_epoch was only advanced inside the
            # use_fast_slow branch, so a normal O1O resume restarted from epoch 0 every time.
            # Restore epoch progress + LR scheduler for the plain O1O (non fast-slow) resume path.
            if not getattr(model_without_ddp, "use_fast_slow", False):
                if "lr_scheduler" in checkpoint and "epoch" in checkpoint:
                    try:
                        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                        # FIX (2026-10-27 v2): torch 2.5.1 StepLR.get_lr() reads
                        # optimizer.param_groups[i]['lr'], NOT base_lrs. The previous stage's
                        # lr leaks in via optimizer.load_state_dict; the restore loop above
                        # re-applies THIS stage's group lrs. The real bug was the argument-form
                        # lr_scheduler.step(args.lr_drop) at the milestone (double-decays and
                        # collapses lr to ~0, observed: task2_ft lr->0 @ep51). Fix: drive the
                        # scheduler with plain no-arg step() in the training loop and align
                        # last_epoch to start_epoch so the decay lands at the intended global
                        # epoch (lr_drop), decaying 2e-4 -> 2e-5 exactly at ep50.
                        lr_scheduler.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
                        # FIX v3: args.start_epoch MUST be set before using it for last_epoch
                        args.start_epoch = checkpoint["epoch"] + 1
                        lr_scheduler.last_epoch = args.start_epoch
                        if checkpoint["epoch"] > args.lr_drop:
                            lr_scheduler.step(checkpoint["epoch"] - args.lr_drop)
                        elif checkpoint["epoch"] == args.lr_drop:
                            lr_scheduler.step(args.lr_drop)
                        print("[resume] plain-O1O: start_epoch -> %d, lr_scheduler restored (ckpt epoch=%d)" % (args.start_epoch, checkpoint["epoch"]))
                    except Exception as _e:
                        print("[resume] WARNING: lr_scheduler restore failed, keeping start_epoch=0:", repr(_e))
                        args.start_epoch = 0
        # Fast-Slow: align Fast adapter to Slow (zero-residual start) after loading.
        if getattr(model_without_ddp, 'use_fast_slow', False):
            model_without_ddp.sync_fast_from_slow()
            # 区分"从 Task-1/vanilla 权重全新启动 Task-2" 与 "续训被中断的 Task-2 ckpt"：
            # 若 ckpt 本身不含 fast_path./slow_path./scm. 键，说明是 Task-1/vanilla 权重，
            # 必须作为全新 Task-2 从 epoch 0 开始（否则会把 Task-1 的 epoch=29 当作 Task-2 起点，
            # 导致 Task-2 只训 10 个 epoch）。只有续训 Task-2 ckpt 才沿用其 epoch/lr_scheduler。
            _fs_in_ckpt = any(k.startswith(_fs_keys) for k in checkpoint['model'].keys())
            if _fs_in_ckpt:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                if checkpoint['epoch'] > args.lr_drop:
                    lr_scheduler.step(checkpoint['epoch'] - args.lr_drop)
                elif checkpoint['epoch'] == args.lr_drop:
                    lr_scheduler.step(args.lr_drop)
            else:
                # 全新 Task-2：仅借 Task-1/vanilla 权重初始化；epoch 与 lr_scheduler 从 0 全新开始。
                args.start_epoch = 0
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.gamma = args.lr_drop_gamma

            log_stats = {'train_lr': lr_scheduler.get_last_lr()[0], 'epoch': args.start_epoch}

            if wandb is not None and utils.is_main_process():
                wandb.log(log_stats)
            
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, dataset_val, device, args.output_dir, args=args)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            
    # ---- FOTG：构建冻结旧模型副本 + 约束模块 ----
    # 触发条件：--fotg_on / --margin_on 任一开启，或 --mffc_f_on 开启（MFFC-F 需要
    # criterion.fotg 提供的 old_model / prev / num_known 与 teacher 前向 _out_old）。
    # MFFC-F-only 运行时（--fotg_on 0，仅 --mffc_f_on 1）请把各 fotg_lam_* 置 0、
    # --fotg_dual 0 --fotg_accum_every 0 --fotg_unk_w 0，使原 fotg 自身约束归零、
    # 仅 MFFC-F 生效。
    if getattr(args, 'fotg_on', 0) or getattr(args, 'margin_on', 0) or getattr(args, 'mffc_f_on', 0):
        import copy as _copy
        from fotg_module import FOTGModule
        _old_model = _copy.deepcopy(model_without_ddp).to(device)
        # 参照系修正：默认 deepcopy 得到的是"本次训练起点"的模型。
        # 一旦 --resume 指向的是非首任务的中间 ckpt（例如本任务的 checkpoint0004），
        # 这个"起点"就已经是被当前任务训过若干 epoch 的状态，
        # 于是 W_old / z_ref / J_box 全部来自它 —— F 约束退化成"别偏离本次起点太远"，
        # 而不再是"保持上一任务的旧类功能"，前几轮已发生的遗忘还会被固化成参照系。
        # 显式给出上一个任务的 ckpt 即可还原正确语义。
        _old_ckpt = str(getattr(args, 'fotg_old_ckpt', '') or '')
        if _old_ckpt:
            _sd = torch.load(_old_ckpt, map_location='cpu', weights_only=False)
            _sd = _sd['model'] if isinstance(_sd, dict) and 'model' in _sd else _sd
            # 兼容可能的 DDP / 包装前缀（'module.' / 'model.'）
            if not all(k in _old_model.state_dict() for k in _sd.keys()):
                _fixed = {}
                for _k, _v in _sd.items():
                    _kk = _k
                    for _p in ('module.', 'model.'):
                        if _kk.startswith(_p):
                            _kk = _kk[len(_p):]
                            break
                    _fixed[_kk] = _v
                _sd = _fixed
            _ms, _us = _old_model.load_state_dict(_sd, strict=False)
            print('[FOTG] old_model 已从上一任务 ckpt 加载: %s' % _old_ckpt)
            print('[FOTG]   loaded=%d  missing=%d  unexpected=%d' % (
                len(_sd) - len(_ms) - len(_us), len(_ms), len(_us)))
            if len(_ms):
                print('[FOTG]   WARNING missing keys (前 5): %s' % list(_ms)[:5])
        else:
            print('[FOTG] old_model = deepcopy(当前模型) -> 参照系为本次训练起点；'
                  '若 --resume 非首任务 ckpt，请用 --fotg_old_ckpt 指定上一任务权重。')
        _cfg = dict(
            dim=int(getattr(args, 'fotg_dim', 256)),
            num_queries=int(getattr(args, 'num_queries', 100)),
            k_sub=int(getattr(args, 'fotg_k_sub', 8)),
            w_threshold=float(getattr(args, 'fotg_w_threshold', 0.0)),
            w_topk=int(getattr(args, 'fotg_w_topk', 10)),
            w_soft=bool(getattr(args, 'fotg_w_soft', 1)),
            eta_box=float(getattr(args, 'fotg_eta_box', 1.0)),
            O_0=float(getattr(args, 'fotg_O_0', 0.5)),
            alpha_sup=float(getattr(args, 'fotg_alpha_sup', 0.1)),
            beta_conf=float(getattr(args, 'fotg_beta_conf', 0.1)),
            O_min=float(getattr(args, 'fotg_O_min', 0.2)),
            O_max=float(getattr(args, 'fotg_O_max', 0.8)),
            eta_pd1=float(getattr(args, 'fotg_eta_pd1', 1.0)),
            min_persistence=float(getattr(args, 'fotg_min_persistence', 0.0)),
            topo_every=int(getattr(args, 'fotg_topo_every', 50)),
            eps_fib=float(getattr(args, 'fotg_eps_fib', 1e-3)),
            eps_ov=float(getattr(args, 'fotg_eps_ov', 1e-2)),
            eps_topo=float(getattr(args, 'fotg_eps_topo', 0.5)),
            lam_fib=float(getattr(args, 'fotg_lam_fib', 0.0)),
            lam_ov=float(getattr(args, 'fotg_lam_ov', 0.0)),
            lam_topo=float(getattr(args, 'fotg_lam_topo', 0.0)),
            lam_route=float(getattr(args, 'fotg_lam_route', 0.0)),
            eps_route=float(getattr(args, 'fotg_eps_route', 0.05)),
            lam_unit_route=float(getattr(args, 'fotg_lam_unit_route', 1.0)),
            dual_lr_route=float(getattr(args, 'fotg_dual_lr_route', 1.0)),
            lam_max_route=float(getattr(args, 'fotg_lam_max_route', 1e3)),
            dual_every=int(getattr(args, 'fotg_dual_every', 20)),
            accum_every=int(getattr(args, 'fotg_accum_every', 20)),
        )
        criterion.fotg = FOTGModule(
            model_without_ddp, _old_model,
            prev_cls=int(getattr(args, 'PREV_INTRODUCED_CLS', 0)),
            cur_cls=int(getattr(args, 'CUR_INTRODUCED_CLS', 20)),
            cfg=_cfg)
        print('[FOTG] 已构建：prev=%d cur=%d eps=%s lam_init=%s' % (
            int(getattr(args, 'PREV_INTRODUCED_CLS', 0)),
            int(getattr(args, 'CUR_INTRODUCED_CLS', 20)),
            {k: _cfg['eps_' + k] for k in ('fib', 'ov', 'topo')},
            {k: _cfg['lam_' + k] for k in ('fib', 'ov', 'topo')}))
    print(f'Start training from epoch {args.start_epoch} to {args.epochs}')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # FIX (2026-10-27 v2): plain no-arg step() every epoch (torch-recommended).
        # The argument-form step(args.lr_drop) double-decayed and froze lr; last_epoch
        # is aligned in the resume block so the decay still lands at args.lr_drop.
        lr_scheduler.step()

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, lr_scheduler=lr_scheduler, args=args, logger=None, wandb=wandb, 
        )
        
        if args.output_dir:
            # 只覆盖最新 checkpoint.pth（不写每轮副本），避免 /dev/shm(tmpfs) 被多轮副本写爆。
            checkpoint_paths = [output_dir / 'checkpoint.pth']

            # 编号 checkpoint 保存频率：默认每 10 轮 + 每 task 最后一轮（用户要求）。
            # 与 eval 频率解耦（eval 仍按 eval_every 做，不再顺带写编号 ckpt）。
            _ckpt_every = int(os.environ.get('FOTG_CKPT_EVERY', '10'))
            _is_last = (args.epochs - epoch) < 1
            _save_numbered = _ckpt_every > 0 and ((epoch + 1) % _ckpt_every == 0 or _is_last)

            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.eval_every == 0 or epoch == 0 or epoch == 1 or (args.epochs-epoch)<1:
                test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, 
                                                  data_loader_val, dataset_val, device, args.output_dir, args=args)
                if _save_numbered and os.environ.get('FOTG_SAVE_EVAL_CKPTS', '0') == '1':
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                if wandb is not None and utils.is_main_process():
                    test_stats["metrics"]['epoch']=epoch
                    wandb.log({str(key): val for key, val in test_stats["metrics"].items()})                
            else:
                 test_stats = {}
                    
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

            # Fast Plasticity: persist memory state & representation buffers
            # (independent of the O1O checkpoint; write under /dev/shm).
            if getattr(model_without_ddp, 'fp', None) is not None:
                if args.fp_memory_save:
                    model_without_ddp.save_fp_memory(args.fp_memory_save)
                    print('[FastPlasticity] saved memory ->', args.fp_memory_save)
                if args.fp_rep_log and args.fp_rep_log_dir:
                    rep_path = os.path.join(args.fp_rep_log_dir,
                                            f'fp_rep_task{args.task_id}_epoch{epoch:04}.pt')
                    if model_without_ddp.save_fp_representation(rep_path):
                        print('[FastPlasticity] saved representation ->', rep_path)
                    model_without_ddp.dump_fp_gate_by_class(
                        os.path.join(args.fp_rep_log_dir,
                                     f'fp_gate_by_class_task{args.task_id}_epoch{epoch:04}.json'))
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if wandb is not None and utils.is_main_process():
            wandb.log(log_stats)
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.dataset in ['owod', 'owdetr'] and (epoch+1) % args.eval_every == 0 and epoch > 0:
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
                            
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.exemplar_replay_selection:
        with torch.no_grad():
            model.eval()        
            image_sorted_scores = get_exemplar_replay(model, exemplar_selection, device, data_loader_train, args)
            create_ft_dataset(args, image_sorted_scores)        
    

def get_datasets(args):
    print(args.train_set)
    print(args.test_set)

    dataset_train = OWDetection(args, 
                    args.data_root, 
                    image_set=args.train_set, 
                    dataset = args.dataset, 
                    pseudo_path=args.pseudo_path,
                    transforms=make_coco_transforms(args.train_set))
    dataset_val = OWDetection(args, 
                    args.data_root, 
                    image_set=args.test_set, 
                    dataset = args.dataset, 
                    transforms=make_coco_transforms(args.test_set))

    print(dataset_train)
    print(dataset_val)

    dataset_val.superclass_names = dataset_train.superclass_names
    dataset_val.superclass_indices = dataset_train.superclass_indices

    return dataset_train, dataset_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)