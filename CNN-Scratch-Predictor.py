import streamlit as st
import numpy as np
import pickle
from PIL import Image
from streamlit_drawable_canvas import st_canvas
def correlate2d(x, k):
    h, w = k.shape
    out_h = x.shape[0] - h + 1
    out_w = x.shape[1] - w + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(x[i:i+h, j:j+w] * k)
    return out

def convolve2d(x, k):
    k = np.flip(np.flip(k, 0), 1)
    h, w = k.shape
    pad_h, pad_w = h - 1, w - 1
    x_padded = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)))
    out = np.zeros(x.shape)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x_padded[i:i+h, j:j+w] * k)
    return out

def sigmoid(x):
        return 1/(1+np.exp(-x))
def sigmoid_der(x):
        return sigmoid(x)*(1-sigmoid(x))

class Layer:
    def forward(self, input):
        pass

    def backward(self, grad_output,lr):
        pass

class Dense(Layer):
    def __init__(self, inp, out):
        self.weight = np.random.randn(out, inp) * 0.1
        self.bias = np.zeros((out, 1))

    def forward(self, inp):
        self.inp = inp
        return np.dot(self.weight, self.inp) + self.bias
    def backward(self, out_grad, lr):
        weight_grad = np.dot(out_grad, self.inp.T)
        inp_grad = np.dot(self.weight.T, out_grad)
        self.weight -= lr * weight_grad
        self.bias -= lr * out_grad
        return inp_grad

class Convolutional_Layer(Layer):
    def __init__(self, inp_shape, kernel_size, depth):
        inp_depth, inp_height, inp_width = inp_shape
        self.depth = depth
        self.inp_shape = inp_shape = inp_shape
        self.inp_depth = inp_depth
        self.out_shape = (depth, inp_height - kernel_size + 1, inp_width - kernel_size +1)
        self.kernels_shape= (depth, inp_depth, kernel_size, kernel_size)
        self.kernels = np.random.rand(*self.kernels_shape)
        self.biases = np.random.rand(*self.out_shape)
    def forward(self,inp):
        self.inp = inp
        self.out = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.inp_depth):
                self.out[i] += correlate2d(self.inp[j], self.kernels[i,j])
        return self.out
    def backward(self, out_grad, lr):
        kernels_grad = np.zeros(self.kernels_shape)
        inp_grad = np.zeros(self.inp_shape)
        for i in range(self.depth):
            for j in range(self.inp_depth):
                kernels_grad[i,j] = correlate2d(self.inp[j], out_grad[i])
                inp_grad[j] += convolve2d(out_grad[i], self.kernels[i,j])
        self.kernels -= lr*kernels_grad
        self.biases -= lr*out_grad
        return inp_grad

class Reshape(Layer):
    def __init__(self, inp, out):
        self.inp_shape = inp
        self.out_shape = out
    def forward(self, inp):
        return np.reshape(inp, self.out_shape)
    def backward(self, out_grad,lr):
        return np.reshape(out_grad, self.inp_shape)

def BCE(y_true, y_pred):
    return -np.mean(y_true*np.log(y_pred) + (1 - y_true)*np.log(1-y_pred))
def BCE_der(y_true, y_pred):
    return ((1-y_true)/(1-y_pred)-y_true/ y_pred)/ np.size(y_true)

class Activatoo(Layer):
    def __init__(self, act, act_der):
        self.act = act
        self.act_der = act_der
    def forward(self, inp):
        self.inp = inp
        return self.act(self.inp)
    def backward(self, out_grad, lr):
        return out_grad * self.act_der(self.inp)

class Sigmoid(Activatoo):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_der)

Convutional_Layer = Convolutional_Layer

with open("cnn_model_trained.pkl", "rb") as f:
    network = pickle.load(f)

st.success("Model loaded successfully !")
st.title("Scratch CNN – Draw & Predict")

canvas = st_canvas(
    stroke_width=9,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

submit = st.button("Submit / Predict")

if submit and canvas.image_data is not None:
    img = canvas.image_data[:, :, :3].astype(np.uint8)

    pil = Image.fromarray(img).convert("L").resize((28,28))
    small = np.array(pil)

    x = small / 255.0
    x = x.reshape(1, 28, 28)

    out = x
    for layer in network:
        out = layer.forward(out)

    class_label = ['A Top','A Bottom']
    prob = out.item()
    pred = int(prob > 0.5)

    st.subheader("Prediction")
    st.write(f"Class: {class_label[pred]}")


