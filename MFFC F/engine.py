# ------------------------------------------------------------------------
# NOTE(2026-10-17): 本文件因磁盘满导致的误操作被清空，后由 __pycache__ 中
# 损坏前版本的 .pyc 反演重建。重建源码经 python3.10 重新编译后与原件字节码
# 递归比对完全一致（co_code/co_consts/co_names/co_varnames/flags/argcount
# 全部相同），行为严格等价；源码注释与排版可能与原件有差异，特此说明。
# ------------------------------------------------------------------------
# O1O: Grouping Known Classes to Identify Unknown Objects as Odd-One-Out
# Misra Yavuz, Fatma Guney. ACCV, 2024. (https://arxiv.org/abs/2410.07514)
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DN-DETR: Accelerate DETR Training by Introducing Query DeNoising
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import numpy as np
from typing import Iterable
import torch
import util.misc as utils
from datasets.open_world_eval import OWEvaluator

try:
    from fast_slow_scm import consolidation_loss
except Exception:
    consolidation_loss = None

# 新 F-B（PCA-directional Jacobian Protection）排除的"读头"参数：
# 这些参数读 hs_last 但不生成它，不属于投影对象 θ。
HEAD_KEYS = ("class_embed", "superclass_embed", "bbox_embed", "enc_class_embed", "enc_bbox_embed")

# PCA 方向基缓存（按 basis_pt 路径缓存，避免每 step 重复读盘）
_PCA_BASIS_CACHE = {}

# 新 F-B 的 J_K 缓存（每 pca_fab_every 步重算一次，中间步复用）
_PCA_FAB_JK_CACHE = None
_PCA_FAB_LAST_STEP = -1


def _get_pca_basis(args):
    """加载预计算 PCA 方向基，返回 (mu [1,D], V_K [D,K])，缓存避免重复读盘。

    basis .pt 应包含键 'mu'（[1,D] 或 [D]）、'V'（[D,64] 全方向）、
    可选 'dirs'（冲突方向索引 list）。--pca_dirs 可覆盖方向选择。
    """
    pt = getattr(args, 'pca_basis_pt', '')
    if not pt:
        return None
    key = (pt, getattr(args, 'pca_dirs', ''))
    if key in _PCA_BASIS_CACHE:
        return _PCA_BASIS_CACHE[key]
    try:
        d = torch.load(pt, map_location='cpu', weights_only=False)
        mu = torch.as_tensor(d['mu'], dtype=torch.float32)
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)                    # [1,D]
        V = torch.as_tensor(d['V'], dtype=torch.float32)   # [D,64]
        dirs = getattr(args, 'pca_dirs', '')
        if dirs:
            ks = [int(x) for x in str(dirs).split(',') if x != '']
        else:
            ks = list(d.get('dirs', [])) if 'dirs' in d else []
        if not ks:
            return None
        V_K = V[:, ks]                              # [D,K]
        # 搬到训练设备
        dev = None
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dev = torch.device('cuda', torch.distributed.get_rank() % torch.cuda.device_count()) if torch.cuda.is_available() else torch.device('cpu')
        except Exception:
            pass
        if dev is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mu = mu.to(dev); V_K = V_K.to(dev)
        _PCA_BASIS_CACHE[key] = (mu, V_K)
        return _PCA_BASIS_CACHE[key]
    except Exception as e:
        print(f"[pca_fab] 加载 PCA 基失败 {pt}: {e}")
        return None

def slprint(x, name='x'):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        print(f'{name}.shape:', x.shape)
    elif isinstance(x, (tuple, list)):
        print('type x:', type(x))
        for i in range(min(10, len(x))):
            slprint(x[i], f'{name}[{i}]')
    elif isinstance(x, dict):
        for k,v in x.items():
            slprint(v, f'{name}[{k}]')
    else:
        print(f'{name}.type:', type(x))


def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k,v in item.items()}
    else:
        raise NotImplementedError("Call Shilong if you use other containers! type: {}".format(type(item)))

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None, wandb=None,
                    ):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    import os as _os
    _fa_probe = None
    if _os.environ.get('FOTG_PROBE', '0') == '1':
        from fotg_fa_probe import FAProbe
        _fa_probe = FAProbe(_os.environ.get('FOTG_PROBE_OUT', '/dev/shm/fa_probe.jsonl'),
                            sample_every=int(_os.environ.get('FOTG_PROBE_EVERY', '20')))
        _mm_probe = model.module if hasattr(model, 'module') else model
        _ce_probe = getattr(_mm_probe, 'class_embed', None)
        if _ce_probe is not None:
            _fa_probe.attach(_ce_probe[-1] if isinstance(_ce_probe, (torch.nn.ModuleList, list)) else _ce_probe)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                dn_args=(targets, args.scalar, args.label_noise_scale, args.box_noise_scale, args.num_patterns)
                if args.contrastive is not False:
                    dn_args += (args.contrastive,)

                outputs, mask_dict = model(samples, dn_args=dn_args)
                loss_dict = criterion(outputs, targets, mask_dict, epoch=epoch)
            else:
                outputs = model(samples)
                loss_dict = criterion(outputs, targets, epoch=epoch)

            fp_stats = model.pop_fp_stats() if hasattr(model, 'pop_fp_stats') else {}
            if fp_stats:
                metric_logger.update(**{f'fp_{k}': v for k, v in fp_stats.items() if isinstance(v, (int, float))})
            if getattr(args, 'use_fast_slow', False) and getattr(args, 'scm_loss_w', 0.0) > 0 and consolidation_loss is not None and hasattr(model, '_scm_info'):
                scm_l = sum(consolidation_loss(i['f_ca'], i['f_fast_sel'], i['M']) for i in model._scm_info)
                loss_dict['scm_loss'] = args.scm_loss_w * scm_l

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            _labels = None
            _z_cur = None
            fab_P = None
            _pca_fab_JK = None   # 新 F-B: J_K = ∂(V_Kᵀ(h-μ))/∂θ，backward 后对 p.grad 投影
            # ---- FOTG：F/O/T/U 四个约束 + 对偶上升（--fotg_on 0 时完全不执行） ----
            if getattr(criterion, 'fotg', None) is not None and 'hs_last' in outputs:
                _fotg = criterion.fotg
                _fotg._iter = getattr(_fotg, '_iter', 0) + 1
                _step = _fotg._iter
                _mf = model.module if hasattr(model, 'module') else model
                # train 态带 DN：hs[-1] 前 pad_size 行是 DN/匹配查询，末 n_reg 行才是
                # 常规 decoder 查询；旧模型 eval 态无 DN（正好 n_reg 行）。两侧都切到
                # 末 n_reg 行，形状对齐（与 150 步因果消融完全同构）。
                # ★batch>1 修复：hs_last 是 [B, pad+n_reg, D]，必须按 batch 维度切
                #   每张图的末 n_reg 行，再 flatten 成 [B*n_reg, D]。
                #   旧写法 reshape(-1,D)[-_n_reg:] 在 batch=2 时只保留第 2 张图
                #   （flatten 后末 n_reg 行恰是最后一张图），且 N=100 ≠ 正确的 200，
                #   与 pred_logits（[B, n_reg, C]，已 dn_post_process 去 DN）错位，
                #   引发 fotg_module.old_scores 的 161 vs 80 崩溃。
                _n_reg = int(getattr(args, 'num_queries', 100)) * max(int(getattr(args, 'num_patterns', 0)), 1)
                _hs = outputs['hs_last']
                if _hs.dim() == 3:      # [B, pad+n_reg, D]
                    _B = _hs.shape[0]
                    _z_cur = _hs.reshape(_B, -1, _hs.shape[-1])[:, -_n_reg:, :].reshape(-1, _hs.shape[-1])
                else:                   # 兼容 [N, D]（无 batch 维的旧路径）
                    _z_cur = _hs.reshape(-1, _hs.shape[-1])[-_n_reg:]
                # 旧模型前向：no_grad + eval，只为了拿 z_ref 与旧门控分数
                with torch.no_grad():
                    if need_tgt_for_training:
                        _oo = _fotg.old_model(samples, dn_args=args.num_patterns)
                    else:
                        _oo = _fotg.old_model(samples)
                    _out_old = _oo[0] if isinstance(_oo, (tuple, list)) else _oo
                    _hs2 = _out_old['hs_last']
                    if _hs2.dim() == 3:
                        _B2 = _hs2.shape[0]
                        _z_ref = _hs2.reshape(_B2, -1, _hs2.shape[-1])[:, -_n_reg:, :].reshape(-1, _hs2.shape[-1])
                    else:
                        _z_ref = _hs2.reshape(-1, _hs2.shape[-1])[-_n_reg:]
                # ---- F-A gate：--fotg_on 0 时 F-A 完全不执行（compute/Jacobian/对偶
                # 上升/几何累积全跳过），仅保留 teacher 前向供 MFFC-F 使用。
                # MFFC-F 不依赖 _c/_labels（只依赖 _out_old/_z_ref/_fotg.prev）。
                _fa_on = bool(getattr(args, 'fotg_on', 0))
                _c, _labels = None, None
                if _fa_on:
                    _c, _parts = _fotg.compute(_z_cur, _z_ref, _out_old, outputs,
                                               targets, _step)
                    _main_loss = losses
                    _losses, _used = _fotg.total_loss(losses, _c, _step)
                    losses = _losses
                # ---- F-A gradient probe (read-only; FOTG_PROBE=1) ----
                if _fa_probe is not None and _fa_on:
                    _fa_loss = losses - _main_loss
                    _pt = {}
                    _ht = _fa_probe.head_input()
                    if _ht is not None and _ht.requires_grad:
                        _pt['hs_head_in'] = _ht
                    _ce = getattr(_mf, 'class_embed', None)
                    if _ce is not None:
                        _pt['class_embed_last'] = (_ce[-1].weight if isinstance(_ce, (torch.nn.ModuleList, list)) else _ce.weight)
                    _se = getattr(_mf, 'superclass_embed', None)
                    if _se is not None:
                        _pt['superclass_embed_last'] = (_se[-1].weight if isinstance(_se, (torch.nn.ModuleList, list)) else _se.weight)
                    _fa_probe.measure(_main_loss, _fa_loss, _pt, epoch, _step)
                # ---- F-B：Jacobian 软投影（事前主动干预更新方向） ----
                # 复用 FOTGModule 缓存的旧类读出雅可比（W_old + J_box），构造每样本
                # 投影矩阵 P = I - Jᵀ(JJᵀ+λI)⁻¹J；在 backward 阶段把特征空间梯度投
                # 影到旧读出雅可比的零空间，压低"易改变旧任务输出"的更新方向，保留
                # 新类/检测方向。Adam 自适应性不受影响（每步只改梯度方向一次）。
                fab_P = None
                if getattr(args, 'fab_on', 0) and _fotg.prev > 0:
                    from fotg_losses import nullspace_projection
                    _J = _fotg.old_readout_jacobian(_z_ref)   # (N, C_old+4, D)
                    if _J is not None:
                        fab_P = nullspace_projection(
                            _J, lam=float(getattr(args, 'fab_lambda', 1e-3)))
                # F-B 投影通过 backward hook 作用于 hs_last 的特征空间梯度：
                # 先 losses.backward() 让所有参数（含 class_embed/superclass_embed/
                # bbox_embed 等检测头）正常拿梯度，hook 在 hs_last.grad 到达时把
                # 末 n_reg 行 query 特征的梯度投影到旧读出雅可比零空间。
                # ★修复：旧实现用 _z_cur.backward(_gzp) 替代 losses.backward()，
                #   导致接在 hs_last 之后的检测头完全拿不到梯度（task2 时 class_embed
                #   权重 30 epoch 一个 bit 都不变，新类 CK=0 的根因）。改用 hook 后，
                #   head 梯度与特征投影二者兼得。
                _fab_hook = None
                if fab_P is not None and _hs is not None and _hs.requires_grad:
                    # ★必须挂在原始 hs_last 上（_z_cur 是 reshape(-1,D) 的 copy，
                    #   挂在其上投影不会回传 transformer）。hook 收到 hs_last 梯度后，
                    #   只对末 n_reg 行（常规 decoder query）做投影，DN 前 pad 行保持原梯度。
                    _P_hook = fab_P.detach()
                    _n_reg_hook = _n_reg
                    def _proj_grad(_g, _P=_P_hook, _nreg=_n_reg_hook):
                        _g = _g.clone()
                        if _g.dim() == 3:
                            _B = _g.shape[0]
                            _greg = _g.reshape(_B, -1, _g.shape[-1])[:, -_nreg:, :]
                            _gproj = torch.einsum('nij,nj->nj', _P, _greg.reshape(-1, _g.shape[-1]))
                            _greg[...] = _gproj.reshape(_B, _nreg, _g.shape[-1])
                        else:
                            _greg = _g.reshape(-1, _g.shape[-1])[-_nreg:]
                            _gproj = torch.einsum('nij,nj->nj', _P, _greg)
                            _greg[...] = _gproj
                        return _g
                    _fab_hook = _hs.register_hook(_proj_grad)
                # ---- 新 F-B：PCA-directional Jacobian Protection（--pca_fab_on 1 时生效） ----
                # 与旧 F-B（nullspace_projection 对特征梯度）独立：新 F-B 对"生成 hs_last 的
                # 全量可训练参数 θ"（排除读头）的梯度 g 做 PCA 冲突方向软投影：
                #   g' = g - J_K^T (J_K J_K^T + λI)^{-1} J_K g,   J_K = ∂(V_K^T(h-μ))/∂θ
                # V_K 为预计算的 PCA 冲突方向基（--pca_basis_pt 加载，--pca_dirs 指定方向）。
                # 只投影冲突方向（避免 blanket 投影 old 敏感方向误伤 new 类学习）。
                # 这里先算 J_K（retain_graph=True 保留图，供后续 losses.backward 用），
                # 投影动作推迟到 backward 之后（见下方 optimizer.step 前）。
                if getattr(args, 'pca_fab_on', 0) and _fotg.prev > 0 and _hs is not None and _hs.requires_grad:
                    global _PCA_FAB_JK_CACHE, _PCA_FAB_LAST_STEP
                    _pca_every = int(getattr(args, 'pca_fab_every', 20))
                    _pca_basis = _get_pca_basis(args)
                    # 每 _pca_every 步重算 J_K；中间步复用缓存（省去 3×全参数 autograd.grad）
                    if _pca_basis is not None and _n_reg > 0 and (
                            _PCA_FAB_JK_CACHE is None or (_step - _PCA_FAB_LAST_STEP) >= _pca_every):
                        _mu = _pca_basis[0]; _VK = _pca_basis[1]   # mu [1,D], V_K [D,K]
                        # 末 n_reg 行（常规 decoder query，与旧 F-B 同口径），带梯度图
                        if _hs.dim() == 3:
                            _Bp = _hs.shape[0]
                            _h_reg = _hs.reshape(_Bp, -1, _hs.shape[-1])[:, -_n_reg:, :].reshape(-1, _hs.shape[-1])
                        else:
                            _h_reg = _hs.reshape(-1, _hs.shape[-1])[-_n_reg:]
                        _qK = (_h_reg - _mu) @ _VK          # [N_reg, K] 带图
                        _qbar = _qK.mean(0)                 # [K]
                        # θ = 生成 hs_last 的全量可训练参数（排除读头）
                        _mf_p = model.module if hasattr(model, 'module') else model
                        _theta = [p for n, p in _mf_p.named_parameters()
                                  if p.requires_grad and not any(k in n for k in HEAD_KEYS)]
                        # J_K = ∂qbar/∂θ ∈ R^{K×P}：对 K 个冲突方向分别求导（每方向一行），
                        # retain_graph=True 保留图供后续 losses.backward 使用。
                        _K = _qbar.shape[0]
                        _JK_rows = []
                        for _k in range(_K):
                            _e = torch.zeros(_K, device=_qbar.device); _e[_k] = 1.0
                            _grads = torch.autograd.grad(_qbar, _theta,
                                    grad_outputs=_e, retain_graph=True, allow_unused=True)
                            _row = torch.cat([
                                (g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=_qbar.device))
                                for g, p in zip(_grads, _theta)])
                            _JK_rows.append(_row)
                        _PCA_FAB_JK_CACHE = torch.stack(_JK_rows, dim=0).detach()   # [K, P]
                        _PCA_FAB_LAST_STEP = _step
                        loss_dict['pca_fab_JK_norm'] = _PCA_FAB_JK_CACHE.norm().detach()
                        loss_dict['pca_fab_refresh'] = torch.tensor(1.0, device=_qbar.device)
                    elif _PCA_FAB_JK_CACHE is not None:
                        loss_dict['pca_fab_refresh'] = torch.tensor(0.0, device=_hs.device)
                    _pca_fab_JK = _PCA_FAB_JK_CACHE
                # 记录约束值与 λ，便于观察"约束是否真的在起作用"（仅 F-A 开时）
                if _c is not None:
                    for _k in ('fib', 'route', 'ov', 'topo'):
                        _v = _c.get(_k)
                        if _v is not None and torch.is_tensor(_v):
                            loss_dict['fotg_C_' + _k] = _v.detach()
                            loss_dict['fotg_lam_' + _k] = torch.as_tensor(
                                _fotg.lam[_k], device=_v.device, dtype=_v.dtype)
                # 对偶上升（不必每步做，间隔 dual_every 步即可；仅 F-A 开时）
                if _fa_on and getattr(args, 'fotg_dual', 1) and _step % _fotg.dual_every == 0:
                    _fotg.dual_step(_c)
                # 几何记忆：把 query 特征按 GT 匹配标签灌进去（U/T 的原料；仅 F-A 开时）
                if _fa_on and _fotg.accum_every > 0 and _step % _fotg.accum_every == 0:
                    _labels = _fotg.match_labels_to_queries(outputs, targets)
                    _fotg.maybe_accumulate(_z_cur, _labels, _step)
                # 模块 U 的可训练项：把未知 query 推出所有已知类切空间（仅 F-A 开时）
                if _fa_on and getattr(args, 'fotg_unk_w', 0.0) > 0 and _labels is not None:
                    _p_obj = torch.ones(_z_cur.shape[0], device=_z_cur.device)
                    _p_max = torch.softmax(
                        outputs['pred_logits'].reshape(_z_cur.shape[0], -1)[:, :80].float(),
                        dim=-1).max(dim=-1).values.detach()
                    _lu, _ = _fotg.compute_unk(_z_cur, _p_obj, _p_max, _labels)
                    loss_dict['fotg_unk'] = args.fotg_unk_w * _lu
                    losses = losses + loss_dict['fotg_unk']

                # ---- MARGIN：Old-New margin 保持损失（--margin_on 0 时跳过） ----
                # 复用本块已算出的教师前向 _out_old（T1 参照）与当前 outputs['pred_logits']（T2）。
                #   m_i^(T) = z_{y_i}^(T) - max_{c∈C_new} z_c^(T)
                #   L_margin = λ·mean max(0, m_i^{T1} - m_i^{T2})
                # 仅对"已匹配到旧类 GT 的 query"生效；同时监控 New GT 的 new-vs-old margin，
                # 防止方法通过压低所有新类 logit 来虚假改善旧类 margin。
                if getattr(args, 'margin_on', 0) and _out_old is not None:
                    _N = _z_ref.shape[0]
                    _zcur = outputs['pred_logits'].reshape(-1, outputs['pred_logits'].shape[-1]).float()
                    if _zcur.shape[0] != _N:
                        _zcur = _zcur[-_N:]
                    _zref = _out_old['pred_logits'].reshape(-1, _out_old['pred_logits'].shape[-1]).float()
                    if _zref.shape[0] != _N:
                        _zref = _zref[-_N:]
                    _lab = _fotg.match_labels_to_queries(outputs, targets)  # (N,) -1=未匹配
                    _prev = int(_fotg.prev); _nknown = int(_fotg.num_known)
                    _old = (_lab >= 0) & (_lab < _prev)
                    # 记录旧类匹配 query 数 (>=1 即 margin 损失可触发), 用于监控 2x2 margin 轴激活情况
                    loss_dict['margin_old_match_n'] = torch.tensor(int(_old.sum().item()), dtype=torch.float32, device=losses.device)
                    if _old.any():
                        _yi = _lab[_old].to(_zcur.device).long()
                        _mT2 = _zcur[_old, _yi] - _zcur[_old, _prev:_nknown].amax(1)
                        _mT1 = _zref[_old, _yi] - _zref[_old, _prev:_nknown].amax(1)
                        _eps = float(getattr(args, 'margin_eps', 0.0))
                        _lm = torch.clamp(_mT1 - _eps - _mT2, min=0.0).mean()
                        losses = losses + float(getattr(args, 'margin_lam', 0.1)) * _lm
                        loss_dict['margin_loss'] = _lm.detach()
                        loss_dict['margin_oldvnew_mean'] = _mT2.mean().detach()
                        loss_dict['margin_teacher_oldvnew_mean'] = _mT1.mean().detach()
                    _new = (_lab >= _prev) & (_lab < _nknown)
                    if _new.any():
                        _yn = _lab[_new].to(_zcur.device).long()
                        _nvold = _zcur[_new, _yn] - _zcur[_new, 0:_prev].amax(1)
                        loss_dict['margin_newvold_mean'] = _nvold.mean().detach()

                # ---- MFFC-F：多通道功能纤维约束（F 模块新版，--mffc_f_on 1 时生效） ----
                # 与 MARGIN / F-A / F-B / PCA-FAB 正交独立：保护旧类 *功能* 而非表征。
                # 仅在"已匹配到旧类 GT 的 query"上计算；旧类缺失时各通道=0（不引入新噪声）。
                # 依赖 criterion.fotg（已在 main 中随 --mffc_f_on 一并构建：提供 old_model /
                # prev / num_known）；teacher 前向 _out_old 已在本 FOTG 块内算好。
                # 三实现边界：① 需旧类输入（无旧 GT 时该块为 0）；② DETR 按 GT 对齐独立匹配
                #   student/teacher 再比较，不依赖固定 query slot；③ 三通道独立归一化后再进 L_fib。
                if getattr(args, 'mffc_f_on', 0) and _out_old is not None and getattr(_fotg, 'prev', 0) > 0:
                    from mffc_f import MFFCFConstraint
                    _mmf = model.module if hasattr(model, 'module') else model
                    if not hasattr(_fotg, '_mffc_f'):
                        _fotg._mffc_f = MFFCFConstraint(
                            topk=int(getattr(args, 'mffc_f_topk', 5)),
                            iou_thr=float(getattr(args, 'mffc_f_iou', 0.5)),
                            lam_id=float(getattr(args, 'mffc_f_lam_id', 1.0)),
                            lam_fg=float(getattr(args, 'mffc_f_lam_fg', 1.0)),
                            lam_loc=float(getattr(args, 'mffc_f_lam_loc', 1.0)),
                            lam_fib=float(getattr(args, 'mffc_f_lam_fib', 0.1)),
                            loc_scale=float(getattr(args, 'mffc_f_loc_scale', 1.0)),
                            fib_w_fg=float(getattr(args, 'mffc_f_fib_w_fg', 1.0)),
                            fib_w_loc=float(getattr(args, 'mffc_f_fib_w_loc', 1.0)),
                            temp=float(getattr(args, 'mffc_f_temp', 1.0)),
                        )
                    _mffc = _fotg._mffc_f
                    _noobj = int(_fotg.num_known)
                    # head_fn 仅 L_fib 需要：z_T -> (cls_logits, bbox)，用当前读头
                    _head_fn = None
                    if float(getattr(args, 'mffc_f_lam_fib', 0.1)) > 0:
                        def _mffc_head_fn(z):
                            _ce = getattr(_mmf, 'class_embed', None)
                            _be = getattr(_mmf, 'bbox_embed', None)
                            if _ce is None or _be is None:
                                _C = int(_fotg.num_known) + 1
                                return (z.new_zeros(z.shape[0], _C), z.new_zeros(z.shape[0], 4))
                            _ce = _ce[-1] if isinstance(_ce, (torch.nn.ModuleList, list)) else _ce
                            _be = _be[-1] if isinstance(_be, (torch.nn.ModuleList, list)) else _be
                            return _ce(z), _be(z)
                        _head_fn = _mffc_head_fn
                    # 形状安全对齐（与 MARGIN 块同口径；正常 pred_logits 已为 [B,n_reg,C]）
                    _Bm = int(outputs['pred_logits'].shape[0])
                    _nreg = int(outputs['pred_logits'].shape[1])
                    def _safe3(t, last):
                        x = t.reshape(-1, last)
                        if x.shape[0] != _z_ref.shape[0]:
                            x = x[-_z_ref.shape[0]:]
                        return x.reshape(_Bm, _nreg, last)
                    _logS = _safe3(outputs['pred_logits'], outputs['pred_logits'].shape[-1])
                    _bxS = _safe3(outputs['pred_boxes'], 4)
                    _logT = _safe3(_out_old['pred_logits'], _out_old['pred_logits'].shape[-1])
                    _bxT = _safe3(_out_old['pred_boxes'], 4)
                    _ls, _parts = _mffc(
                        _z_cur, _z_ref, _logS, _logT, _bxS, _bxT,
                        targets, int(_fotg.prev), int(_fotg.num_known), _noobj,
                        head_fn=_head_fn,
                    )
                    _mw = float(getattr(args, 'mffc_f_w', 1.0))
                    losses = losses + _mw * _ls
                    for _kk in ('L_id', 'L_fg', 'L_loc', 'L_fib', 'n_pairs'):
                        _v = _parts.get(_kk)
                        if _v is not None:
                            loss_dict['mffc_' + _kk] = _v.detach()

            if getattr(args, 'use_fast_slow', False) and getattr(args, 'scm_loss_w', 0.0) > 0 and 'scm_loss' in loss_dict:
                losses = losses + loss_dict['scm_loss']

            if getattr(args, 'sep_loss_w', 0.0) > 0:
                _m = model.module if hasattr(model, 'module') else model
                _ce = getattr(_m, 'class_embed', None)
                if isinstance(_ce, torch.nn.ModuleList):
                    _ce = _ce[-1]
                if _ce is not None:
                    _W = _ce.weight
                    _prev = int(getattr(args, 'PREV_INTRODUCED_CLS', 0))
                    _cur = int(getattr(args, 'CUR_INTRODUCED_CLS', 0))
                    if _prev > 0 and _cur > 0 and _prev + _cur <= _W.shape[0]:
                        _UP = _W[:_prev]
                        _UC = _W[_prev:_prev + _cur]
                        _UP = _UP / (_UP.norm(dim=1, keepdim=True) + 1e-8)
                        _UC = _UC / (_UC.norm(dim=1, keepdim=True) + 1e-8)
                        _cross = _UP @ _UC.t()
                        _sep = (_cross ** 2).sum() / float(_prev * _cur)
                        loss_dict['sep_loss'] = args.sep_loss_w * _sep
                        losses = losses + loss_dict['sep_loss']


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            # F-B 投影已通过 _z_cur 上的 backward hook 完成（见上），统一标准反向。
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            # F-B 投影已通过 _z_cur 上的 backward hook 完成（见上），这里统一走
            # 标准 losses.backward()，保证所有检测头都拿到梯度。
            optimizer.zero_grad()
            losses.backward()
            # ---- 新 F-B：backward 后对 θ 梯度做 PCA 方向软投影（--pca_fab_on 1） ----
            if _pca_fab_JK is not None:
                from fotg_losses import pca_directional_projection
                _mf_p = model.module if hasattr(model, 'module') else model
                _theta = [p for n, p in _mf_p.named_parameters()
                          if p.requires_grad and not any(k in n for k in HEAD_KEYS)]
                _resid = pca_directional_projection(
                    _pca_fab_JK, _theta, lam=float(getattr(args, 'pca_fab_lam', 1e-3)))
                if _resid is not None:
                    loss_dict['pca_fab_resid'] = _resid
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            if getattr(args, 'use_fast_slow', False) and hasattr(model, 'fs_consolidate'):
                model.fs_consolidate(getattr(args, 'ema_rho', 0.01))


        if wandb is not None and utils.is_main_process():
            wandb.log({"total_loss":loss_value})
            wandb.log(loss_dict_reduced_scaled)
            wandb.log(loss_dict_reduced_unscaled)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None,
             eval_mode=False, eval_threshold=0.0):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args)

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # import ipdb; ipdb.set_trace()
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:

                if hasattr(args, 'cascade_attn') and args.cascade_attn:
                    outputs, _ = model(samples, dn_args=(targets,args.num_patterns))
                else:
                    outputs, _ = model(samples, dn_args=args.num_patterns)

            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, eval_mode=eval_mode, eval_threshold=eval_threshold)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # import ipdb; ipdb.set_trace()
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        
        if args.save_results:
            """
            saving results of eval.
            """
            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()[1]
        
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stats['metrics']=res
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # import ipdb; ipdb.set_trace()
    return stats, coco_evaluator

    
@torch.no_grad()
def get_exemplar_replay(model, exemplar_selection, device, data_loader, args):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[ExempReplay]'
    print_freq = 10

    image_sorted_scores_reduced={}
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, _ = model(samples, dn_args=args.num_patterns)
            else:
                outputs = model(samples)

        image_sorted_scores = exemplar_selection(samples, outputs, targets)
        for i in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(i[0])
            
        metric_logger.update(loss=len(image_sorted_scores_reduced.keys()))
        
    print(f'found a total of {len(image_sorted_scores_reduced.keys())} images')
    return image_sorted_scores_reduced


def create_ft_dataset(args, image_sorted_scores):
    print(f'found a total of {len(image_sorted_scores.keys())} images')

    tmp_dir=args.data_root +'/ImageSets/'+args.dataset+"/"+str(args.exemplar_replay_dir)+"/"

    class_sorted_scores={}
    imgs_per_class={}
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS):
        class_sorted_scores[str(i)]=[]
        imgs_per_class[str(i)]=[]

    for k,v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            if v['labels'][j] >= args.PREV_INTRODUCED_CLS and v['labels'][j] < args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS:
                class_sorted_scores[str(v['labels'][j])].append(v['scores'][j])

    class_threshold={}
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS):
        tmp=np.array(class_sorted_scores[str(i)])
        tmp.sort()
        tmp = torch.Tensor(tmp)
        if len(tmp)>args.num_inst_per_class and not args.exemplar_replay_random:
            max_val = tmp[-args.num_inst_per_class//2]
            min_val = tmp[args.num_inst_per_class//2]
        else:
            if args.exemplar_replay_random:
                print('using random exemplar selection')
            else:
                print(f'only found {len(tmp)} imgs in class {i}')
            max_val = tmp.min()
            min_val = tmp.max()
            
        class_threshold[str(i)]=(min_val, max_val)

    save_imgs = []    
    for k,v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            if v['labels'][j] >= args.PREV_INTRODUCED_CLS and v['labels'][j] < args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS:                
                label = str(v['labels'][j])
                if (v['scores'][j] <= class_threshold[label][0].numpy() or v['scores'][j] >= class_threshold[label][1].numpy()) and (len(imgs_per_class[label])<=args.num_inst_per_class+2):
                    img = str(k)[4:]
                    if len(img)==10:
                        imgs_per_class[label].append(img[:4]+'_'+img[4:])
                        save_imgs.append(img[:4]+'_'+img[4:])
                    else:
                        save_imgs.append(img)
                        imgs_per_class[label].append(img)

                        
    print(f'found {len(np.unique(save_imgs))} images in run')

    if len(args.exemplar_replay_prev_file)>0:
        previous_ft = open(tmp_dir+args.exemplar_replay_prev_file,'r').read().splitlines()
        save_imgs+=previous_ft
        
    save_imgs=np.unique(save_imgs)
    np.random.shuffle(save_imgs)
    if len(save_imgs)> args.exemplar_replay_max_length:
        save_imgs=save_imgs[:args.exemplar_replay_max_length]
    
    os.makedirs(tmp_dir, exist_ok=True)
    print('saving to:', tmp_dir+args.exemplar_replay_cur_file)
    with open(tmp_dir+args.exemplar_replay_cur_file, 'w') as f:
        for line in save_imgs:
            f.write(line)
            f.write('\n')

