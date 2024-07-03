# 최종 오류
/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/autograd/graph.py:744: UserWarning: Error detected in ReluBackward0. Traceback of forward call that caused the error:
  File "/home/safeai24/safe24/pytorch_GAN_zoo/AiS/arbitrary_image_stylization_jieun/src/train_t.py", line 95, in <module>
    main(parse_args())
  File "/home/safeai24/safe24/pytorch_GAN_zoo/AiS/arbitrary_image_stylization_jieun/src/train_t.py", line 64, in main
    loss = net_with_loss(content, style)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/safeai24/safe24/pytorch_GAN_zoo/AiS/arbitrary_image_stylization_jieun/src/train_t.py", line 30, in forward
    stylized = self.network(content, style)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/safeai24/safe24/pytorch_GAN_zoo/AiS/arbitrary_image_stylization_jieun/src/model/ais_t.py", line 34, in forward
    stylized_images = self.transform((content, self.norm, style_params))
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/safeai24/safe24/pytorch_GAN_zoo/AiS/arbitrary_image_stylization_jieun/src/model/transform_t.py", line 107, in forward
    out = self.expand(x)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/safeai24/safe24/pytorch_GAN_zoo/AiS/arbitrary_image_stylization_jieun/src/model/transform_t.py", line 71, in forward
    x = self.conv((x, normalizer_fn, params, order))
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/safeai24/safe24/pytorch_GAN_zoo/AiS/arbitrary_image_stylization_jieun/src/model/transform_t.py", line 34, in forward
    x = self.activation_fn(x)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/modules/activation.py", line 103, in forward
    return F.relu(input, inplace=self.inplace)
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/nn/functional.py", line 1500, in relu
    result = torch.relu(input)
 (Triggered internally at ../torch/csrc/autograd/python_anomaly_mode.cpp:111.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  0%|                                                                                                                                                                | 0/4 [00:43<?, ?it/s]
Traceback (most recent call last):
  File "/home/safeai24/safe24/pytorch_GAN_zoo/AiS/arbitrary_image_stylization_jieun/src/train_t.py", line 95, in <module>
    main(parse_args())
  File "/home/safeai24/safe24/pytorch_GAN_zoo/AiS/arbitrary_image_stylization_jieun/src/train_t.py", line 65, in main
    loss.backward()
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/_tensor.py", line 525, in backward
    torch.autograd.backward(
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/autograd/__init__.py", line 267, in backward
    _engine_run_backward(
  File "/home/safeai24/miniconda3/envs/zoo/lib/python3.9/site-packages/torch/autograd/graph.py", line 744, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [8, 3, 128, 128]], which is output 0 of ReluBackward0, is at version 2; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
