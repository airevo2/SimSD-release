# CONTRACT — moe_dispatch (LLaDA2 MoE 专家派发)

**mode**: `inference`（forward-only，无 autograd.Function）
**gpu**: NVIDIA RTX PRO 6000 Blackwell Server Edition, **sm_120**, 96 GB
**torch** 2.8.0+cu128 · **triton** 3.4.0
**建立于** 2026-07-27

---

## 1. 为什么做

实测单次 mini 前向 50.0 ms，其中 **32.5 ms (65%)** 是专家派发；把专家循环换成同形状
no-op 后前向降到 17.5 ms。每次前向约 **1311 个极小 Linear 调用 + 19 次 host 同步**
（19 个 MoE 层 × 中位 23 个活跃专家 × 3 个 Linear）。

这同时是 cuda_graph 不可用的原因：`moe_infer` 里有
`tokens_per_expert.cpu().numpy()` 与 `.item()`。见 `docs/llada2-plan.md`。

## 2. 参考实现

`LLaDA2MoeSparseMoeBlock.moe_infer`，HF remote code
`inclusionAI/LLaDA2.0-{mini,flash}`（`$HF_MODULES_CACHE/transformers_modules/`）。
**不改 thirdparty 源码** —— 通过 `kernels/fused_toggle.py` 猴补，由
`speculative_decoding/adapters/llada2.py` 驱动（与既有 thin_linear 的做法一致）。

### 前向数学

对 token `t` 及其 top-k 专家 `e_j`（权重 `w_j`）：

```
h_j   = silu(x_t @ Wg[e_j].T) * (x_t @ Wu[e_j].T)      # (I,)
y_j   = h_j @ Wd[e_j].T                                 # (H,)
out_t = ( Σ_j  fp32(y_j) * w_j ).to(x.dtype)            # 归约在 fp32
```

**dtype 纪律**（照抄参考实现，不可简化）：
- `topk_weight` 恒为 **fp32**（gate 里 `F.linear` 强制 fp32 输入 → logits fp32）
- 专家输出先 `.type(topk_weight.dtype)` 即转 fp32，乘权重、按 k 求和，**最后**才
  `.type(new_x.dtype)` 转回 x 的 dtype
- `silu` = `ACT2FN["silu"]`，`hidden_act="silu"`

### 隔离后的纯签名

```python
moe_dispatch(x, topk_ids, topk_weight, w_gate, w_up, w_down) -> Tensor
    x:           (T, H)     bf16 | fp32     已 flatten 的 hidden_states
    topk_ids:    (T, k)     int64           gate 选出的专家下标
    topk_weight: (T, k)     fp32            归一化并乘过 routed_scaling_factor
    w_gate:      (E, I, H)  x.dtype         堆叠后的 gate_proj.weight
    w_up:        (E, I, H)  x.dtype         堆叠后的 up_proj.weight
    w_down:      (E, H, I)  x.dtype         堆叠后的 down_proj.weight
    ->           (T, H)     x.dtype
```

注意 `nn.Linear.weight` 是 `(out, in)`，所以 `w_gate[e]` 是 `(I, H)`、
`w_down[e]` 是 `(H, I)`；矩阵乘要用 `.T`。

**不在本算子范围内**：gate 本身、`shared_experts`、`num_shared_experts` 的加法。
它们留在原 `forward` 里。

## 3. 形状

| | E | top-k | H | I | T（每块） |
|---|---|---|---|---|---|
| mini | 256 | 8 | 2048 | 512 | 32 |
| flash | 256 | 8 | 4096 | 1024 | 32 |

`M_total = T * k = 256`（静态）。活跃专家实测中位 **23**（全 MASK 输入）/
**43**（真实内容），远小于 E=256 —— 路由高度集中，这是设计的关键约束。

verify 前向的 T 是 `n_known + 2*n_masks` ≤ `2*block_length`，故 M_total ≤ 512。

## 4. ⚠️ 验收判据（已修正）

**原先提的「fp32 下与 stock 逐位相同 max|Δ|==0」在原理上不可能达成**，
因为本算子就是要把 per-expert 的 cuBLAS GEMM 换成 grouped GEMM —— 归约顺序必然改变。
逐位相同只适用于不换 kernel 的改动（例如之前验证分片时）。

改用仓库既定容差（skill Conventions）：

| dtype | 前向容差 |
|---|---|
| fp32 | `< 1e-5` |
| bf16 | `2**-7 * ref.abs().max() + 1e-3` |

**补充一条端到端判据**（比逐元素容差更贴近我们真正在意的东西）：
fp32 下，整模型 logits 的 **argmax 逐 token 相同**。理由是已知 LLaDA2 的 MoE 路由对
数值极敏感（bf16 下 ~1/21 的 token 会翻转专家），所以 bf16 的 token 级比较只作记录、
不作判据；fp32 的 argmax 一致性才是有牙的。

## 5. CUDA-graph 安全约束（本 mode 的硬要求）

这是做这件事的**首要目的**，不是附带项：

- ❌ 不得有 host 同步：`.item()` / `.cpu()` / `.tolist()` / `int(tensor)` / 数据依赖的
  Python 分支
- ❌ 不得有数据依赖的**形状**：所有中间张量形状必须由 `(T, k, E, H, I)` 静态决定
- ❌ 热路径内不得分配（capture 期分配的地址会被冻结；预分配或依赖 caching allocator
  的稳定复用）
- ✅ 允许数据依赖的**数值**（例如 `offs` 的内容），只要形状静态
- Triton grid 必须静态 → 按最坏情况 `min(E, M_total)` 个 M-tile 起，空 tile 早退

管线实际使用情况：`--pipeline` 与 `fused_denoise` 走 `torch.cuda.graph` capture；
torch.compile **未使用**，故不需要 `torch.library.custom_op` 包装。

## 6. 设计（correctness-first）

```
1. flat = topk_ids.view(-1)                        # (M_total,)
2. order = flat.argsort(stable=True)               # GPU sort
3. rows  = order // k                              # 每个 slot 对应的 token 行
4. counts = bincount(flat, minlength=E)            # (E,)
5. 按 BLOCK_M 对齐：每个活跃专家补齐到 BLOCK_M 的整数倍
   -> sorted_slots (M_pad,)  expert_per_tile (NUM_TILES,)   均为静态形状
6. grouped GEMM ×3（或 gate/up 融合）：
      g = A @ Wg[e].T ; u = A @ Wu[e].T ; hh = silu(g)*u ; y = hh @ Wd[e].T
7. unsort + fp32 加权求和 -> (T, H)
```

`M_pad = M_total + E*(BLOCK_M-1)` 是最坏界；实际只有活跃专家产生 tile，空 tile 早退。

## 7. 状态

- [x] Step 1 workspace
- [x] Step 2 定位与隔离
- [x] Step 3 oracle — 23 case（synthetic / extreme / **real 权重**）
- [x] Step 4 kernel — `kernels/moe_dispatch_fused.py`
- [x] Step 5 对拍 — **36/36 通过**，baseline 已记账
- [x] Step 6 toggle — `fused_toggle.apply_moe_to_model()`
- [ ] Step 7 report.html

## 8. 结果

| 项 | 值 |
|---|---|
| 对拍 | **36/36**（23 持久 + 13 现生成随机/边界） |
| 整模型单次前向（mini, 32tok, bf16, 真实内容） | **78.19 → 27.90 ms = 2.80×**（no-op 地板 ~17.5 ms） |
| fp32 端到端 argmax | **32/32 一致**，相对误差 1.6e-6 |
| bf16 端到端 argmax | 26/32 —— 与已知 MoE 路由敏感度同带，非回归 |
| **cuda_graph** | 单算子 **capture + replay 成功**；换路由后 replay 数值正确 |
| bf16 精度 vs fp64 | fused 是 stock 误差的 **0.90–1.20×**（5 个 case 中 2 个更准） |

### 过程中修正的三件事

1. **验收判据本身是错的。** 「fp32 逐位相同」对换 kernel 的改动原理上不可能达成。
   判别实验证明：同样分组、只把 grouped GEMM 换成逐组 cuBLAS，与 oracle `max|Δ|=0`
   —— 分组/gather/fp32 归约逻辑完全正确，差异只来自 GEMM 累加顺序。
2. **容差写成了绝对值。** fp32 的 `<1e-5` 是绝对界而 bf16 的是相对界；`large_mag`
   绝对误差 1.95e-2 但相对只有 4.9e-7。改为相对。
3. **`torch.bincount` 在 CUDA 上会同步**，单这一处就足以让本算子进不了 cuda graph。
   换成 `scatter_add_`。

### 已知限制（交给 kernel-optimize）

- **Triton 的 fp32 `tl.dot` 精度约为 cuBLAS 的 4–7× ULP**（隔离实测：K=2048 时
  26 ULP vs 3.9 ULP），且与 `BLOCK_K` 无关（64/128/256 结果一致）。
  部署 dtype 是 bf16（实测 1.00×），故不阻塞；但 fp32 验证路径的裕度因此变窄。
- `BLOCK_M=16` 而活跃专家平均只有 ~3 个 slot → tile 内约 80% 是 padding。
  这是「把 ~250 次 launch 压成 4 次」的代价，也是 kernel-optimize 的首要目标。
- gate/up 未融合，激活是独立的 elementwise kernel。
