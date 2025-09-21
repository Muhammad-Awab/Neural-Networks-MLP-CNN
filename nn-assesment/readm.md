### make_blobs functtion
<!-- The function creates datapoints (random coordinates).
Then it splits them into two groups:
Left blob = class 0 → label 0
Right blob = class 1 → label 1
So you end up with:
X → the points (features the model will use).
y → the class labels (the answers the model should learn).
👉 That way, when we train the neural network, it can learn:
“Points on the left are class 0, points on the right are class 1.” -->


### class MLP  (Input → Hidden Layer 1 → ReLU → Hidden Layer 2 → ReLU → Output → Logit)
<!-- MLP is a neural network model inheriting from nn.Module, which is the base class for all PyTorch models.

This means it can:

Store layers as attributes.

Automatically track parameters for optimization.

Define a forward method to compute outputs from inputs. -->

<!-- forward defines how data moves through the network.
forward defines how data flows through the network.
x is a batch of input features: shape [batch_size, in_dim].
self.net(x) applies all layers sequentially.
.squeeze(1) removes the extra dimension from the output:
nn.Linear(..., 1) gives shape [batch_size, 1].
squeeze(1) converts it to [batch_size].-->

### Data Class
<!-- 3️⃣ Advantages of @dataclass
Less boilerplate – no need to manually define __init__ or __repr__.
Clear structure – makes your hyperparameters look like a clean, immutable “record”.
Easy to modify defaults – can easily create variants: -->
<!-- 
A normal class can be used to define entities with behavior, logic, and custom methods. A data class focuses on storing and processing data with minimal code -->

# Learning Rate
<!-- Learning rate for the optimizer (0.01).

Controls how big a step the model takes while updating weights. -->



![alt text](image.png)





### nn.Conv2d(1, 32, kernel_size=3, padding=1),  
<!-- What are these 32 shapes?

Each little 3×3 square is one kernel (filter) that your CNN learned automatically during training.

The network doesn’t know digits at first.

It only knows: “I have 32 little 3×3 windows that I can adjust.”

Training adjusts these tiny windows so that they become pattern detectors.

What patterns do they detect?

At this first layer, they are very simple:

Vertical edges (dark on left, light on right)

Horizontal edges (dark on top, light on bottom)

Diagonals (slanted strokes)

Blobs (spots of light/dark)

So each kernel specializes in looking for one type of feature anywhere in the image.

Why do we need 32 different ones?

Because a digit has many possible local features:

The vertical bar of “1”

The loop in “0”

The diagonal of “7”

The horizontal top of “5”

One filter can’t detect all of them.
👉 So the CNN learns a whole set of 32 filters, each focusing on different simple patterns. -->