# Export from a Model Checkpoint from MXNet and Into MMS

## Using Your Own Trained Models and Checkpoints

While all of these features are super exciting you've probably been asking yourself, so how do I create these fabulous MMS model files for my own trained models? We'll provide some MXNet code examples for just this task.

There are two main routes for this: 1) export a checkpoint or use the new `.export` function, or 2) using a MMS Python class to export your model directly.

The Python method to export model is to use `export_serving` function while completing training:

```python
   import mxnet as mx
   from mms.export_model import export_serving

   mod = mx.mod.Module(...)
   # Training process
   ...

   # Export model
   signature = { "input_type": "image/jpeg", "output_type": "application/json" }
   export_serving(mod, 'resnet-18', signature, aux_files=['synset.txt'])
```

In MXNet with version higher than 0.12.0, you can export a Gluon model directly, as long as your model is Hybrid:

```python
from mxnet import gluon
net = gluon.nn.HybridSequential() # this mode will allow you to export the model

with net.name_scope():
    net.add(gluon.nn.Dense(128, activation="relu")) # an example first layer
    # Add the rest of your network architecture here

net.hybridize() # hybridize your network so that it can be exported

# Then train your network before moving on to exporting

signature = {
                "input_type": "application/json",
                "inputs" : [
                    {
                        "data_name": "data",
                        "data_shape": [1, 100]
                    }
                ],
                "outputs" : [
                    {
                        "data_name": "softmax",
                        "data_shape": [1, 128]
                    }
                ],
                "output_type": "application/json"
            }

export_serving(net, 'gluon_model', signature, aux_files=['synset.txt'])
```

**Note**: be careful with versions. If you export a v0.12 model and try to run it with MMS running v0.11 of MXNet, the server will probably throw errors and you won't be able to use the model.
