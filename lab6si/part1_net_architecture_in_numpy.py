import numpy as np
from data import LinearlySeparableClasses, NonlinearlySeparableClasses
from visualization_utils import inspect_data, plot_data, x_data_from_grid, visualize_activation_function, \
    plot_two_layer_activations



def relu(logits):
    return np.maximum(logits, 0)

def sigmoid(logits):
    return 1. / (1. + np.exp(-logits))

def hardlim(logits):
    return (logits > 0).astype(np.float32)

def linear(logits):
    return logits


def zad1_single_neuron(student_id):
    gen = LinearlySeparableClasses()
    x, y = gen.generate_data(seed=student_id)
    n_samples, n_features = x.shape

    class SingleNeuron:
        def __init__(self, n_in, f_act):
            self.W = 0.01 * np.random.randn(n_in, 1)  
            self.b = 0.01 * np.random.randn(1)  
            self.f_act = f_act

        def forward(self, x_data):
            z = np.dot(x_data, self.W) + self.b
            return self.f_act(z)

    model = SingleNeuron(n_in=n_features, f_act=hardlim)

    model.W[:, 0] = [-1, -1]
    model.b[:] = [0.4]

    y_pred = model.forward(x)
    print(f'Accuracy = {np.mean(y == y_pred) * 100}%')

    x_grid = x_data_from_grid(min_xy=-1, max_xy=2, grid_size=1000)
    y_pred_grid = model.forward(x_grid)
    plot_data(x, y, plot_xy_range=[-1, 2], x_grid=x_grid, y_grid=y_pred_grid, title='Linia decyzyjna neuronu')


def zad2_two_layer_net(student_id):
    gen = NonlinearlySeparableClasses()
    x, y = gen.generate_data(seed=student_id)
    n_samples, n_features = x.shape

    class DenseLayer:
        def __init__(self, n_in, n_out, f_act):
            self.W = 0.01 * np.random.randn(n_in, n_out)  
            self.b = 0.01 * np.random.randn(n_out) 
            self.f_act = f_act

        def forward(self, x_data):
            z = x_data @ self.W + self.b
            return self.f_act(z)

    class SimpleTwoLayerNetwork:
        def __init__(self, n_in, n_hidden, n_out):
            self.hidden_layer = DenseLayer(n_in=n_in, n_out=n_hidden, f_act=relu)
            self.output_layer = DenseLayer(n_in=n_hidden, n_out=n_out, f_act=hardlim)

        def forward(self, x_data):
            h = self.hidden_layer.forward(x_data)
            out = self.output_layer.forward(h)
            return out

    model = SimpleTwoLayerNetwork(n_in=n_features, n_hidden=2, n_out=1)
    model.hidden_layer.W[:, 0] = [-1.2, 1]
    model.hidden_layer.W[:, 1] = [1.2, -1]
    model.hidden_layer.b[:] = [-0.1, -0.4]
    model.output_layer.W[:, 0] = [1, 1]
    model.output_layer.b[:] = [-0.5]

    y_pred = model.forward(x)
    print(f'Accuracy = {np.mean(y == y_pred) * 100}%')
    plot_two_layer_activations(model, x, y)


if __name__ == '__main__':

    student_id = 123321

    zad1_single_neuron(student_id)
    zad2_two_layer_net(student_id)
