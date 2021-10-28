"""
Runs the Huggingface BertForSequenceClassification model using the Lazy Tensor Core with the TorchScript backend.

Requirements to run example:
- `transformers` Python package by HuggingFace
- `torchvision` Python package
- `pillow` Python package
- `requests` Python package
- `lazy_tensor_core` Python package
    For information on how to obtain the `lazy_tensor_core` Python package,
    see here:

    https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/QUICKSTART.md

To run the example, make sure `/path/to/pytorch/lazy_tensor_core` is in your
PYTHONPATH. Then, run

    python lazytensor_resnet18_example.py

The output of this example can be found in
    `lazytensor_resnet18_example_output.txt`

Most of the code in this example was barrowed from
    https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_resnet18_example.py
    https://github.com/llvm/torch-mlir/blob/main/build_tools/torchscript_e2e_heavydep_tests/minilm_sequence_classification.py
"""

import torch
import lazy_tensor_core as ltc
from lazy_tensor_core.debug import metrics

from torch._C import CompilationUnit
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend
from torch_mlir.passmanager import PassManager

from utils.annotator import Annotation
from utils.torch_mlir_types import TorchTensorType
# from lazytensor.builder import build_module

ltc._LAZYC._ltc_init_ts_backend()

DEVICE = 'lazy'

# Initialize HuggingFace transformers
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def _prepare_sentence_tokens(sentence: str):
    return torch.tensor([tokenizer.encode(sentence)])

class HFBertModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", 
            num_labels=
            2,  # The number of output labels--2 for binary classification.
            output_attentions=
            False,  # Whether the model returns attentions weights.
            output_hidden_states=
            False,  # Whether the model returns all hidden-states.
            torchscript=True).to(DEVICE)
        self.bert.eval()

    # @export
    # @annotate_args([
    #     None,
    #     ([-1, -1], torch.int64, True),
    # ])
    def forward(self, tokens):
        return self.bert.forward(tokens)[0]


def main():

    # Create dummy text input
    test_input = _prepare_sentence_tokens("this project is very interesting").to(DEVICE)

    bert_module = HFBertModule()
    print('Running bert.forward...')
    result = bert_module.forward(test_input)

    print('\nMetrics report:')
    print(metrics.metrics_report())
    graph_str = ltc._LAZYC._get_ltc_tensors_backend([bert_module.forward(test_input)])
    print(graph_str)

    # Create a torch.jit.ScriptFunction out of the graph
    cu = CompilationUnit()
    func_name = 'my_method'
    script_function = cu.create_function(func_name, graph_str)

if __name__ == '__main__':
    main()
