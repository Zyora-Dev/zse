  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 31, in _compile_kernels
    raise _COMPILE_ERROR
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 287, in forward
    output = int4_matmul(x, self.weight_packed, self.weight_scales, self.group_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 217, in int4_matmul
    return Int4MatmulFunction.apply(input, weight, scales, group_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 188, in forward
    kernel = _compile_kernels()
             ^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/kernels/int4_matmul.py", line 170, in _compile_kernels
    raise _COMPILE_ERROR
RuntimeError: Failed to compile INT4 kernels: Error building extension 'zse_int4_matmul': [1/3] c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=zse_int4_matmul -DTORCH_API_INCLUDE_EXTENSION_H -isystem /home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include -isystem /home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda-12.4/include -isystem /usr/include/python3.11 -fPIC -std=c++17 -c /home/ionet/.cache/torch_extensions/py311_cu128/zse_int4_matmul/main.cpp -o main.o 
FAILED: [code=1] main.o 
c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=zse_int4_matmul -DTORCH_API_INCLUDE_EXTENSION_H -isystem /home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include -isystem /home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda-12.4/include -isystem /usr/include/python3.11 -fPIC -std=c++17 -c /home/ionet/.cache/torch_extensions/py311_cu128/zse_int4_matmul/main.cpp -o main.o 
In file included from /home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include/torch/csrc/Device.h:4,
                 from /home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include/torch/csrc/api/include/torch/python.h:8,
                 from /home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include/torch/extension.h:9,
                 from /home/ionet/.cache/torch_extensions/py311_cu128/zse_int4_matmul/main.cpp:1:
/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include/torch/csrc/python_headers.h:12:10: fatal error: Python.h: No such file or directory
   12 | #include <Python.h>
      |          ^~~~~~~~~~
compilation terminated.
[2/3] /usr/local/cuda-12.4/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=zse_int4_matmul -DTORCH_API_INCLUDE_EXTENSION_H -isystem /home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include -isystem /home/ionet/benchmark_env/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda-12.4/include -isystem /usr/include/python3.11 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -O3 --use_fast_math -arch=sm_80 -std=c++17 -c /home/ionet/.cache/torch_extensions/py311_cu128/zse_int4_matmul/cuda.cu -o cuda.cuda.o 
/home/ionet/.cache/torch_extensions/py311_cu128/zse_int4_matmul/cuda.cu(69): warning #177-D: variable "warp_id" was declared but never referenced
      int warp_id = tid / 32;
          ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/home/ionet/.cache/torch_extensions/py311_cu128/zse_int4_matmul/cuda.cu(70): warning #177-D: variable "lane_id" was declared but never referenced
      int lane_id = tid % 32;
          ^

ninja: build stopped: subcommand failed.


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 16, in <module>
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/generation/utils.py", line 2566, in generate
    result = decoding_method(
             ^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/generation/utils.py", line 2786, in _sample
    outputs = self(**model_inputs, return_dict=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/utils/generic.py", line 918, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 449, in forward
    outputs: BaseModelOutputWithPast = self.model(
                                       ^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/utils/generic.py", line 1072, in wrapper
    outputs = func(self, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 384, in forward
    hidden_states = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 249, in forward
    hidden_states = self.mlp(hidden_states)
                    ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 46, in forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                                           ^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 295, in forward
    weight = dequantize_int4_zse(
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/ionet/benchmark_env/lib/python3.11/site-packages/zse/format/reader_v2.py", line 69, in dequantize_int4_zse
    dequantized = (unpacked_grouped.float() * scales_expanded).view(out_features, in_features)
                   ~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 260.00 MiB. GPU 0 has a total capacity of 139.72 GiB of which 114.69 MiB is free. Including non-PyTorch memory, this process has 139.60 GiB memory in use. Of the allocated memory 138.14 GiB is allocated by PyTorch, and 824.86 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
(benchmark_env) ionet@iocloud:~$ 
