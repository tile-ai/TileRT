"""Core operations for deepseek v3.2."""

from tilert.models.deepseek_v3_2.ops.down_allreduce import (
    DownAllReduce,
    down_allreduce,
    down_allreduce_glm5,
)
from tilert.models.deepseek_v3_2.ops.eh_proj_allreduce import EHProjAllReduce, eh_proj_allreduce
from tilert.models.deepseek_v3_2.ops.expert_down_allreduce import (
    ExpertDownAllReduce,
    expert_down_allreduce,
)
from tilert.models.deepseek_v3_2.ops.expert_sel_up_gate_silu import (
    ExpertSelectUpGateSiLU,
    ExpertSelectUpGateSiLUAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.expert_select import expert_select
from tilert.models.deepseek_v3_2.ops.flash_sparse_mla import flash_sparse_mla
from tilert.models.deepseek_v3_2.ops.layernorm_rope_rotate import layernorm_rope_rotate
from tilert.models.deepseek_v3_2.ops.projo_wkvb import projo_wkvb
from tilert.models.deepseek_v3_2.ops.projq_wqb import projq_wqb
from tilert.models.deepseek_v3_2.ops.projx_wis import projx_wis
from tilert.models.deepseek_v3_2.ops.qkv_rope import (
    QKVRoPE,
    QKVRoPERefWeightsAlias,
    QKVRoPETilertWeightsAlias,
    qkv_rope,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_expert_proj import RMSNormExpertProj
from tilert.models.deepseek_v3_2.ops.rmsnorm_head_proj import RMSNormHeadProj
from tilert.models.deepseek_v3_2.ops.rmsnorm_kv import rmsnorm_kv
from tilert.models.deepseek_v3_2.ops.rmsnorm_projq_wqib import (
    RmsnormProjqWqib,
    RmsnormProjqWqibAlgorithm,
    RmsnormProjqWqibWeightsConverter,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_projx_wqkvia import (
    RMSNormProjxWqkvia,
    RMSNormProjxWqkviaAlgorithm,
    projx_wqkvia,
    rmsnorm_projx_wqkvia,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_quant import rmsnorm_quant
from tilert.models.deepseek_v3_2.ops.rmsnorm_up_gate_silu import (
    RMSNormUpGateSiLU,
    RMSNormUpGateSiLUAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.rotate import (
    Rotate,
    RotateRefWeightsAlias,
    RotateTilertWeightsAlias,
    rotate,
    rotate_activation,
)
from tilert.models.deepseek_v3_2.ops.sparse_index import sparse_index, sparse_index_topk
from tilert.models.deepseek_v3_2.ops.topk import TopK, topk_accurate, topk_approximate
from tilert.models.deepseek_v3_2.ops.unproj_o_allreduce import (
    UnProjOAllReduce,
    UnProjOAllReduceAlgorithm,
    unproj_o_allreduce,
)
from tilert.models.deepseek_v3_2.ops.up_gate_silu import up_gate_silu

__all__ = [
    "down_allreduce",
    "down_allreduce_glm5",
    "DownAllReduce",
    "expert_down_allreduce",
    "ExpertDownAllReduce",
    "expert_select",
    "up_gate_silu",
    "rmsnorm_projx_wqkvia",
    "projx_wqkvia",
    "rmsnorm_kv",
    "unproj_o_allreduce",
    "projo_wkvb",
    "projq_wqb",
    "rotate",
    "rotate_activation",
    "Rotate",
    "RotateRefWeightsAlias",
    "RotateTilertWeightsAlias",
    "layernorm_rope_rotate",
    "TopK",
    "topk_approximate",
    "topk_accurate",
    "sparse_index",
    "sparse_index_topk",
    "flash_sparse_mla",
    "projx_wis",
    "qkv_rope",
    "QKVRoPE",
    "QKVRoPERefWeightsAlias",
    "QKVRoPETilertWeightsAlias",
    "eh_proj_allreduce",
    "rmsnorm_quant",
    "RmsnormProjqWqib",
    "RmsnormProjqWqibAlgorithm",
    "RmsnormProjqWqibWeightsConverter",
    "RMSNormExpertProj",
    "RMSNormProjxWqkvia",
    "RMSNormProjxWqkviaAlgorithm",
    "RMSNormUpGateSiLU",
    "UnProjOAllReduce",
    "UnProjOAllReduceAlgorithm",
    "RMSNormHeadProj",
    "ExpertSelectUpGateSiLU",
    "ExpertSelectUpGateSiLUAlgorithm",
]
