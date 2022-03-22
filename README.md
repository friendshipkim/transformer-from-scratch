# Transformer-from-scratch

Replication Project: Transformer model built from scratch.

# Setup
```angular2html
pip install requirements.txt
pip install -e .
```

# Checklist
- [x] Check the number of parameters
- [x] Check if output tensors are the same size
- [x] Check self/cross attention masks
    * In torch.transformer, masks are given as inputs. They have the different shape as my model. But checked if baseline masks can be made by reshaping my masks
- [x] Check model.state_dict()
    - [x] Check if the parameters has the same dimensions
- [x] Copy the weights and
    - [x] feed an input and check if the output is the same
        - [x] Without padding
           - [x] FIXED - Precision issue - true when the absolute tolerance is 1e-6, but false when 1e-8
        - [x] With padding
- [x] * Call loss.backward() once and check if the gradients are the same for all parameters

# Loss trends
* baseline
```
    Epoch: 1, Train loss: 4.487, Val loss: 3.824, Epoch time = 60.442s
    Epoch: 2, Train loss: 3.629, Val loss: 3.481, Epoch time = 57.315s
    Epoch: 3, Train loss: 3.318, Val loss: 3.261, Epoch time = 55.679s
    Epoch: 4, Train loss: 3.075, Val loss: 3.107, Epoch time = 52.929s
    Epoch: 5, Train loss: 2.871, Val loss: 2.953, Epoch time = 52.998s
    Epoch: 6, Train loss: 2.663, Val loss: 2.764, Epoch time = 52.787s
    Epoch: 7, Train loss: 2.426, Val loss: 2.615, Epoch time = 52.687s
    Epoch: 8, Train loss: 2.183, Val loss: 2.353, Epoch time = 54.114s
    Epoch: 9, Train loss: 1.957, Val loss: 2.218, Epoch time = 52.905s
    Epoch: 10, Train loss: 1.772, Val loss: 2.134, Epoch time = 52.933s
    Epoch: 11, Train loss: 1.622, Val loss: 2.097, Epoch time = 56.216s
    Epoch: 12, Train loss: 1.495, Val loss: 2.057, Epoch time = 52.930s
    Epoch: 13, Train loss: 1.389, Val loss: 2.056, Epoch time = 53.334s
    Epoch: 14, Train loss: 1.292, Val loss: 2.039, Epoch time = 52.807s
    Epoch: 15, Train loss: 1.208, Val loss: 2.052, Epoch time = 52.755s
```

* my model
```
    Epoch: 1, Train loss: 4.870, Val loss: 4.093, Epoch time = 56.626s
    Epoch: 2, Train loss: 3.876, Val loss: 3.730, Epoch time = 56.522s
    Epoch: 3, Train loss: 3.583, Val loss: 3.519, Epoch time = 56.529s
    Epoch: 4, Train loss: 3.382, Val loss: 3.407, Epoch time = 59.857s
    Epoch: 5, Train loss: 3.213, Val loss: 3.305, Epoch time = 56.343s
    Epoch: 6, Train loss: 3.077, Val loss: 3.218, Epoch time = 56.429s
    Epoch: 7, Train loss: 2.960, Val loss: 3.123, Epoch time = 56.438s
    Epoch: 8, Train loss: 2.854, Val loss: 3.065, Epoch time = 66.608s
    Epoch: 9, Train loss: 2.754, Val loss: 3.006, Epoch time = 56.615s
    Epoch: 10, Train loss: 2.658, Val loss: 2.972, Epoch time = 56.588s
    Epoch: 11, Train loss: 2.566, Val loss: 2.927, Epoch time = 56.805s
    Epoch: 12, Train loss: 2.479, Val loss: 2.896, Epoch time = 56.729s
    Epoch: 13, Train loss: 2.401, Val loss: 2.844, Epoch time = 65.113s
    Epoch: 14, Train loss: 2.321, Val loss: 2.846, Epoch time = 56.403s
    Epoch: 15, Train loss: 2.246, Val loss: 2.825, Epoch time = 56.412s
```