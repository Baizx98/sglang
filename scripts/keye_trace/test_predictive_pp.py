"""Check that shadow selection state survives the actual PP proxy interface."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.keye_topk_mask_moe import KeyeTopKMaskMoeModel, Qwen3MoeModel


class PipelineBoundaryTest(unittest.TestCase):
    def model(self, first):
        obj=KeyeTopKMaskMoeModel.__new__(KeyeTopKMaskMoeModel)
        torch.nn.Module.__init__(obj)
        obj.pp_group=SimpleNamespace(is_first_rank=first)
        return obj

    def test_previous_attention_input_and_exact_ids_cross_boundary(self):
        h=torch.randn(1,2048,dtype=torch.bfloat16)
        ids=torch.arange(2048,dtype=torch.int32).unsqueeze(0)
        first_batch=SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda:True),
            _keye_lookahead_previous_x=h,_keye_lookahead_previous_indices=ids)
        proxy=PPProxyTensors(dict(hidden_states=torch.zeros_like(h),residual=torch.ones_like(h)))
        with patch.dict(os.environ,KEYE_LOOKAHEAD_TRACE_DIR='/unused'), \
             patch.object(Qwen3MoeModel,'forward',return_value=proxy):
            actual=self.model(True).forward(None,None,first_batch)
        second_batch=SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda:True))
        def consume(_self,_ids,_positions,batch,*args,**kwargs):
            self.assertIs(batch._keye_lookahead_previous_x,h)
            self.assertIs(batch._keye_lookahead_previous_indices,ids)
            self.assertIs(kwargs['pp_proxy_tensors'],proxy)
            return torch.tensor(1)
        with patch.dict(os.environ,KEYE_LOOKAHEAD_TRACE_DIR='/unused'), \
             patch.object(Qwen3MoeModel,'forward',new=consume):
            self.model(False).forward(None,None,second_batch,pp_proxy_tensors=actual)

    def test_non_decode_does_not_reuse_stale_selection(self):
        batch=SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda:False))
        with patch.dict(os.environ,KEYE_LOOKAHEAD_TRACE_DIR='/unused'), \
             patch.object(Qwen3MoeModel,'forward',return_value=torch.tensor(1)):
            self.model(False).forward(None,None,batch,pp_proxy_tensors=PPProxyTensors({}))
        self.assertFalse(hasattr(batch,'_keye_lookahead_previous_x'))


if __name__=='__main__':unittest.main()
