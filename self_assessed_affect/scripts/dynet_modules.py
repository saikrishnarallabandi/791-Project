import dynet as dy
import numpy as np


class SequenceVariationalAutoEncoder(object):

  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

   # Label embeddings
    self.num_embed = 5
    self.lookup = model.add_lookup_parameters((self.num_out, self.num_embed))
   
   # LSTM Parameters
    self.lstm_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)  

   # MLP parameters
    num_hidden_q = self.num_latent
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_mean_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_var_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"

    self.W_sm_p = model.add_parameters((num_output, num_hidden))
    self.b_sm_p = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calc_loss_basic(self, frames , label):

    # Renew the computation graph
    dy.renew_cg()

    # Initialize LSTM
    init_state_src = self.lstm_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
       
    input_frames = dy.inputTensor(frames)
    output_label = label

    # Get the LSTM embeddings
    src_output = init_state_src.add_inputs([frame for frame in input_frames])[-1].output()

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_sm, W_sm, z])
    label_embedding = self.lookup[label]
    #print label, label_embedding
    recons_loss = dy.pickneglogsoftmax(pred, label)

    return kl_loss, recons_loss

  def predict_label(self, frames):

    # Renew the computation graph
    dy.renew_cg()

    # Initialize LSTM
    init_state_src = self.lstm_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    input_frames = dy.inputTensor(frames)

    src_output = init_state_src.add_inputs([frame for frame in input_frames])[-1].output()

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    pred = dy.affine_transform([b_sm, W_sm, z])
    return dy.softmax(pred)


class FeedForwardNeuralNet(object):

  def __init__(self, model, args):
    self.pc = model.add_subcollection()
    self.args = args
    self.num_input = int(args[0])
    self.num_output = int(args[2])
    self.hidden_list = args[1]
    self.act = args[3]
    self.model = model
    self.number_of_layers = len(self.hidden_list)
    num_hidden_1 = self.hidden_list[0]
    
    # Add first layer
    self.W1 = self.pc.add_parameters((num_hidden_1, self.num_input))
    self.b1 = self.pc.add_parameters((num_hidden_1))
    
    # Add remaining layers
    self.weight_matrix_array = []
    self.biases_array = []
    self.weight_matrix_array.append(self.W1)
    self.biases_array.append(self.b1)
    for k in range(1, self.number_of_layers):
              self.weight_matrix_array.append(self.model.add_parameters((self.hidden_list[k], self.hidden_list[k-1])))
              self.biases_array.append(self.model.add_parameters((self.hidden_list[k])))
    self.weight_matrix_array.append(self.model.add_parameters((self.num_output, self.hidden_list[-1])))
    self.biases_array.append(self.model.add_parameters((self.num_output)))
    self.spec = (self.num_input, self.hidden_list, self.num_output, self.act)
   
  def basic_affine(self, exp):
    W1 = dy.parameter(self.W1)
    b1 = dy.parameter(self.b1)
    return dy.tanh(dy.affine_transform([b1,W1,exp]))

  def calculate_loss(self, input, output):
    #dy.renew_cg()
    weight_matrix_array = []
    biases_array = []
    for (W,b) in zip(self.weight_matrix_array, self.biases_array):
         weight_matrix_array.append(dy.parameter(W))
         biases_array.append(dy.parameter(b)) 
    acts = self.act
    w = weight_matrix_array[0]
    b = biases_array[0]
    act = acts[0]
    intermediate = act(dy.affine_transform([b, w, input]))
    activations = [intermediate]
    for (W,b,g) in zip(weight_matrix_array[1:], biases_array[1:], acts[1:]):
        pred = g(dy.affine_transform([b, W, activations[-1]]))
        activations.append(pred)  
    losses = output - pred
    return dy.l2_norm(losses)

  def calculate_loss_classification(self, input, output):
    #dy.renew_cg()
    weight_matrix_array = []
    biases_array = []
    for (W,b) in zip(self.weight_matrix_array, self.biases_array):
         weight_matrix_array.append(dy.parameter(W))
         biases_array.append(dy.parameter(b))
    acts = self.act
    w = weight_matrix_array[0]
    b = biases_array[0]
    act = acts[0]
    intermediate = act(dy.affine_transform([b, w, input]))
    activations = [intermediate]
    for (W,b,g) in zip(weight_matrix_array[1:], biases_array[1:], acts[1:]):
        pred = g(dy.affine_transform([b, W, activations[-1]]))
        activations.append(pred)
    losses = dy.pickneglogsoftmax(pred, output)
    return losses



  def predict(self, input):
    weight_matrix_array = []
    biases_array = []
    acts = []
    for (W,b, act) in zip(self.weight_matrix_array, self.biases_array, self.act):
         weight_matrix_array.append(dy.parameter(W))
         biases_array.append(dy.parameter(b))
         acts.append(act)
    g = acts[0]
    w = weight_matrix_array[0]
    b = biases_array[0]
    intermediate = g(w*input + b)
    activations = [intermediate]
    for (W,b, act) in zip(weight_matrix_array[1:], biases_array[1:], acts):
        pred =  act(W * activations[-1]  + b)
        activations.append(pred)
    return pred


class EmbeddingVariationalAutoEncoder(object):
# This takes an utterance level embedding as input as opposed to sequence level input 
  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    self.dnn = FeedForwardNeuralNet(model, [num_input, [num_hidden, num_hidden], num_hidden, [dy.rectify, dy.rectify, dy.rectify]])
    print "Loaded params"

   # Label embeddings
    self.num_embed = 5
    self.lookup = model.add_lookup_parameters((self.num_out, self.num_embed))

   # MLP parameters
    num_hidden_q = self.num_latent
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_mean_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_var_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"

    self.W_sm_p = model.add_parameters((num_output, num_hidden))
    self.b_sm_p = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calc_loss_basic(self, embedding , label):

    # Renew the computation graph
    dy.renew_cg()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
       
    input_embedding = dy.inputTensor(embedding)
    output_label = label

    # Get the LSTM embeddings
    src_output = self.dnn.predict(input_embedding)

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    # Calculate the reconstruction loss
    pred = dy.selu(dy.affine_transform([b_sm, W_sm, z]))
    label_embedding = self.lookup[label]
    recons_loss = dy.pickneglogsoftmax(pred, label)

    return kl_loss, recons_loss

  def predict_label(self, embedding):

    # Renew the computation graph
    dy.renew_cg()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    input_embedding = dy.inputTensor(embedding)

    # Get the DNN encoding
    src_output = self.dnn.predict(input_embedding)

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    pred = dy.affine_transform([b_sm, W_sm, z])
    return dy.softmax(pred)


