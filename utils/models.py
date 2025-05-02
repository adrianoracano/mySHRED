import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from IPython.display import clear_output as clc
from .processdata import mse, mre, num2p

import torch.nn.functional as F
from .processdata import TimeSeriesDataset, Padding

# import numpy as np
# naca0012_data = np.load("I:/Il mio Drive/my_SHRED-ROM/naca0012_data.npz")
# naca0012_airfoil_coords = naca0012_data["naca0012_airfoil_coords"]

import math

class SHRED(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size = 64, hidden_layers = 2, decoder_sizes = [350, 400], dropout = 0.0, 
                 nfeat = 3, mix_dim = 16,
                 ):
        '''
        SHRED model definition
        
        
        Inputs
        	input size (e.g. number of sensors)
        	output size (e.g. full-order variable dimension)
        	size of LSTM hidden layers (default to 64)
        	number of LSTM hidden layers (default to 2)
        	list of decoder layers sizes (default to [350, 400])
        	dropout parameter (default to 0)
        '''
            
        super(SHRED,self).__init__()

        # # if input_size % nfeat != 0:
        # #     raise RuntimeError("Inconsistent features number")
        # self.mix = torch.nn.Linear(nfeat, mix_dim)
        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = hidden_size,
                                  num_layers = hidden_layers,
                                  batch_first=True)
        
        self.decoder = torch.nn.ModuleList()
        decoder_sizes.insert(0, hidden_size)
        decoder_sizes.append(output_size)

        for i in range(len(decoder_sizes)-1):
            self.decoder.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            if i != len(decoder_sizes)-2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    def forward(self, x):

        B, T, nsensors  = x.size()
        # # 1) reshape to isolate per-sensor features
        # x = x.view(B, T, -1, self.mix.in_features)  # -> [B, T, nsensors, nfeat]

        # 2) apply mixing MLP to each sensor's feature vector
        #    will broadcast over B, T, nsensors
        # x = self.mix(x)  # -> [B, T, nsensors, mix_dim]

        # 3) flatten sensors back into feature axis
        # x = x.view(B, T, -1)  # -> [B, T, nsensors * mix_dim]
        
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (output, _) = self.lstm(x, (h_0, c_0))
        output = output[-1].view(-1, self.hidden_size)

        for layer in self.decoder:
            output = layer(output)

        return output

    def freeze(self):

        self.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):

        self.train()
        
        for param in self.parameters():
            param.requires_grad = True

import torch.nn as nn

class SHREDagnostic(nn.Module):
    def __init__(self,
                 coord_dim,
                 latent_dim,
                 output_size,
                 hidden_size=64,
                 lstm_layers=2,
                 decoder_sizes=None,
                 dropout=0.0):
        """
        SHRED model with sensor-agnostic encoding and flexible decoding:
        - if query_coords is provided, does coordinate-based field reconstruction
        - if query_coords is None, directly outputs POD coefficients of size `output_size`
        """
        super(SHREDagnostic, self).__init__()
        # Per-sensor temporal encoder
        self.per_sensor_lstm = nn.LSTM(input_size=1,
                                       hidden_size=hidden_size,
                                       num_layers=lstm_layers,
                                       batch_first=True)
        # MLP for sensor coordinates
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        # Local embedding
        self.local_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, latent_dim),
            nn.ReLU(inplace=True)
        )
        # Direct POD-coeff decoder head
        mlp_out = []
        in_dim = latent_dim
        for hidden in (decoder_sizes or [latent_dim]):
            mlp_out.append(nn.Linear(in_dim, hidden))
            mlp_out.append(nn.ReLU(inplace=True))
            if dropout > 0:
                mlp_out.append(nn.Dropout(dropout))
            in_dim = hidden
        mlp_out.append(nn.Linear(in_dim, output_size))
        self.coef_decoder = nn.Sequential(*mlp_out)
        
    def forward(self, sensor_hist, sensor_coords):
        # sensor_hist: (B, N_s, L)
        B, N_s, L = sensor_hist.size()
        hist = sensor_hist.view(B * N_s, L, 1)
        _, (h_n, _) = self.per_sensor_lstm(hist)
        h_seq = h_n[-1].view(B, N_s, -1)
        coord_emb = self.coord_mlp(sensor_coords.view(B * N_s, -1))
        coord_emb = coord_emb.view(B, N_s, -1)
        local = torch.cat([h_seq, coord_emb], dim=-1)
        local_emb = self.local_mlp(local)
        global_h = local_emb.mean(dim=1)
        # Direct output of POD coefficients
        return self.coef_decoder(global_h)  # (B, output_size)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        self.train()
        for param in self.parameters():
            param.requires_grad = True

class AttentionPooling(nn.Module):
    """
    Attention-based pooling for a set of local embeddings with sensor-level dropout.
    Produces a weighted sum over the N_s embeddings, dropping entire sensor embeddings during training.
    """
    def __init__(self, embed_dim, attn_dim=None, sensor_dropout=0.0):
        super().__init__()
        attn_dim = attn_dim or embed_dim
        self.W = nn.Linear(embed_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.sensor_dropout_p = sensor_dropout

    def forward(self, local_emb):
        # local_emb: (B, N_s, embed_dim)
        B, N_s, D = local_emb.shape
        if self.training and self.sensor_dropout_p > 0:
            mask = local_emb.new_empty(B, N_s, 1).bernoulli_(1 - self.sensor_dropout_p)
            mask = mask / (1 - self.sensor_dropout_p)
            local_emb = local_emb * mask
        e = torch.tanh(self.W(local_emb))        # (B, N_s, attn_dim)
        e = self.v(e).squeeze(-1)                # (B, N_s)
        alpha = F.softmax(e, dim=1)              # (B, N_s)
        global_emb = torch.einsum('bn,bnd->bd', alpha, local_emb)
        return global_emb, alpha

class GraphSensorGNN(nn.Module):
    """
    Simple GNN that refines local embeddings based on sensor coordinates.
    """
    def __init__(self, coord_dim, embed_dim, num_layers=1, sigma=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.sigma = sigma
        self.msg_mlp = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(embed_dim + coord_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, local_emb, coords):
        # local_emb: (B, N_s, D), coords: (B, N_s, coord_dim)
        B, N_s, D = local_emb.shape
        for _ in range(self.num_layers):
            # compute pairwise distances
            diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N_s, N_s, coord_dim)
            dist2 = (diff**2).sum(-1)                          # (B, N_s, N_s)
            A = torch.exp(-dist2 / (2*self.sigma**2))         # (B, N_s, N_s)
            # message computation
            h_i = local_emb.unsqueeze(2).expand(B, N_s, N_s, D)
            h_j = local_emb.unsqueeze(1).expand(B, N_s, N_s, D)
            msgs = torch.cat([h_i, h_j], dim=-1)              # (B,N_s,N_s,2D)
            msgs = self.msg_mlp(msgs)                         # (B,N_s,N_s,D)
            # aggregate weighted by A
            agg = torch.einsum('bij,bijd->bid', A, msgs)      # (B, N_s, D)
            # update embeddings via coords and aggregated messages
            update_input = torch.cat([agg, coords], dim=-1)   # (B, N_s, D+coord_dim)
            delta = self.update_mlp(update_input)             # (B, N_s, D)
            local_emb = local_emb + delta                     # residual update
        return local_emb

class SHREDagnosticAttention(nn.Module):
    def __init__(self,
                 coord_dim,
                 latent_dim,
                 output_size,
                 hidden_size=64,
                 lstm_layers=2,
                 decoder_sizes=None,
                 dropout=0.0,
                 attn_dim=None,
                 sensor_dropout=0.0,
                 use_gnn=False,
                 gnn_layers=1,
                 gnn_sigma=1.0,
                 use_attention=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        # per-sensor LSTM
        self.per_sensor_lstm = nn.LSTM(input_size=1,
                                       hidden_size=hidden_size,
                                       num_layers=lstm_layers,
                                       batch_first=True)
        # coord MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        # local embedding MLP
        self.local_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, latent_dim),
            nn.ReLU(inplace=True)
        )
        # optional GNN
        if use_gnn:
            self.gnn = GraphSensorGNN(coord_dim=coord_dim,
                                      embed_dim=latent_dim,
                                      num_layers=gnn_layers,
                                      sigma=gnn_sigma)
        # pooling
        if use_attention:
            self.attn_pool = AttentionPooling(latent_dim, attn_dim, sensor_dropout=sensor_dropout)
        # decoder
        mlp_out = []
        in_dim = latent_dim
        for h in (decoder_sizes or [latent_dim]):
            mlp_out.append(nn.Linear(in_dim, h)); mlp_out.append(nn.ReLU(inplace=True))
            if dropout>0: mlp_out.append(nn.Dropout(dropout))
            in_dim = h
        mlp_out.append(nn.Linear(in_dim, output_size))
        self.coef_decoder = nn.Sequential(*mlp_out)

    def forward(self, sensor_hist, sensor_coords, query_coords=None):
        B, N_s, L = sensor_hist.size()
        # 1) LSTM
        hist = sensor_hist.view(B * N_s, L, 1)
        _, (h_n, _) = self.per_sensor_lstm(hist)
        h_seq = h_n[-1].view(B, N_s, -1)
        # 2) coord embedding
        coord_emb = self.coord_mlp(sensor_coords.view(B*N_s, -1)).view(B, N_s, -1)
        # 3) local embedding
        local = torch.cat([h_seq, coord_emb], dim=-1)
        local_emb = self.local_mlp(local)
        # 4) optional GNN refinement
        if self.use_gnn:
            local_emb = self.gnn(local_emb, sensor_coords)
        # 5) pooling
        if self.use_attention:
            global_h, attn_weights = self.attn_pool(local_emb)
        else:
            attn_weights = None
            global_h = local_emb.mean(dim=1)
        # 6) decode
        output = self.coef_decoder(global_h)
        return output, attn_weights

    def freeze(self):
        self.eval()
        for p in self.parameters(): p.requires_grad=False

    def unfreeze(self):
        self.train()
        for p in self.parameters(): p.requires_grad=True

# class SHREDagnosticAttention(nn.Module):
#     def __init__(self,
#                  coord_dim,
#                  latent_dim,
#                  output_size,
#                  hidden_size=64,
#                  lstm_layers=2,
#                  decoder_sizes=None,
#                  dropout=0.0,
#                  attn_dim=None,
#                  sensor_dropout=0.0):
#         """
#         SHRED model with sensor-agnostic encoding and attention-based pooling with sensor dropout:
#         - Uses per-sensor LSTM + coord MLP + local embedding
#         - Aggregates local embeddings via attention instead of mean pooling
#         - Drops entire sensor embeddings during training for robustness
#         """
#         super().__init__()
#         # Per-sensor temporal encoder
#         self.per_sensor_lstm = nn.LSTM(input_size=1,
#                                        hidden_size=hidden_size,
#                                        num_layers=lstm_layers,
#                                        batch_first=True)
#         # MLP for sensor coordinates
#         self.coord_mlp = nn.Sequential(
#             nn.Linear(coord_dim, hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(inplace=True)
#         )
#         # Local embedding
#         self.local_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_size, latent_dim),
#             nn.ReLU(inplace=True)
#         )
#         # Attention pooling over sensors with dropout
#         self.attn_pool = AttentionPooling(latent_dim, attn_dim, sensor_dropout=sensor_dropout)
#         # Direct POD-coeff decoder head
#         mlp_out = []
#         in_dim = latent_dim
#         for hidden in (decoder_sizes or [latent_dim]):
#             mlp_out.append(nn.Linear(in_dim, hidden))
#             mlp_out.append(nn.ReLU(inplace=True))
#             if dropout > 0:
#                 mlp_out.append(nn.Dropout(dropout))
#             in_dim = hidden
#         mlp_out.append(nn.Linear(in_dim, output_size))
#         self.coef_decoder = nn.Sequential(*mlp_out)

#     def forward(self, sensor_hist, sensor_coords, query_coords=None):
#         # sensor_hist: (B, N_s, L)
#         B, N_s, L = sensor_hist.size()
#         # 1) LSTM over temporal history
#         hist = sensor_hist.view(B * N_s, L, 1)
#         _, (h_n, _) = self.per_sensor_lstm(hist)
#         h_seq = h_n[-1].view(B, N_s, -1)  # (B, N_s, hidden_size)
#         # 2) Coord embedding
#         coord_emb = self.coord_mlp(sensor_coords.view(B * N_s, -1))
#         coord_emb = coord_emb.view(B, N_s, -1)
#         # 3) Local embedding
#         local = torch.cat([h_seq, coord_emb], dim=-1)  # (B, N_s, 2*hidden)
#         local_emb = self.local_mlp(local)               # (B, N_s, latent_dim)
#         # 4) Attention pooling with sensor dropout
#         global_h, attn_weights = self.attn_pool(local_emb)
#         # 5) Decode POD coefficients
#         output = self.coef_decoder(global_h)            # (B, output_size)
#         return output, attn_weights  # return weights if needed for analysis
    
#     def freeze(self):
#         self.eval()
#         for param in self.parameters():
#             param.requires_grad = False

#     def unfreeze(self):
#         self.train()
#         for param in self.parameters():
#             param.requires_grad = True

def fit(model, train_dataset, valid_dataset, batch_size = 64, epochs = 4000, optim = torch.optim.Adam, lr = 1e-3, loss_fun = mse, loss_output = mre, formatter = num2p, verbose = False, patience = 5):
    ''' 
    Neural networks training
    
    Inputs
    	model (`torch.nn.Module`)
    	training dataset (`torch.Tensor`)
    	validation dataset (`torch.Tensor`)
    	batch size (default to 64)
   	    number of epochs (default to 4000)
    	optimizer (default to `torch.optim.Adam`)
    	learning rate (default to 0.001)
        loss function (defalut to Mean Squared Error)
        loss value to print and return (default to Mean Relative Error)
        loss formatter for printing (default to percentage format)
    	verbose parameter (default to False) 
    	patience parameter (default to 5)
    '''

    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    optimizer = optim(model.parameters(), lr = lr)

    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, epochs + 1):

        # print("epoch ", epoch)
               
        for k, data in enumerate(train_loader):
            # print(f"batch {k}")
            model.train()
            def closure():
                outputs = model(data[0])
                optimizer.zero_grad()
                loss = loss_fun(outputs, data[1])
                loss.backward()
                return loss
            optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            train_error = loss_output(train_dataset.Y, model(train_dataset.X))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)

        if verbose == True:
            print("Epoch "+ str(epoch) + ": Training loss = " + formatter(train_error_list[-1]) + " \t Validation loss = " + formatter(valid_error_list[-1]), flush = True)

        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            model.load_state_dict(best_params)
            train_error = loss_output(train_dataset.Y, model(train_dataset.X))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            
            if verbose == True:
                print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error))
         
            return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
    
    model.load_state_dict(best_params)
    train_error = loss_output(train_dataset.Y, model(train_dataset.X))
    valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
    
    if verbose == True:
        print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error))
        
    return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
 
def fit_sensors_coords(model, train_dataset, valid_dataset, sensors_coords_train = None, sensors_coords_valid = None, batch_size = 64, epochs = 4000, optim = torch.optim.Adam, lr = 1e-3, loss_fun = mse, loss_output = mre, formatter = num2p, verbose = False, patience = 5):
    ''' 
    Neural networks training
    
    Inputs
    	model (`torch.nn.Module`)
    	training dataset (`torch.Tensor`)
    	validation dataset (`torch.Tensor`)
    	batch size (default to 64)
   	    number of epochs (default to 4000)
    	optimizer (default to `torch.optim.Adam`)
    	learning rate (default to 0.001)
        loss function (defalut to Mean Squared Error)
        loss value to print and return (default to Mean Relative Error)
        loss formatter for printing (default to percentage format)
    	verbose parameter (default to False) 
    	patience parameter (default to 5)
    '''
    optimizer = optim(model.parameters(), lr = lr)
    if sensors_coords_train != None:
        train_loader = DataLoader(list(zip(sensors_coords_train, train_dataset.X, train_dataset.Y)), shuffle = True, batch_size = batch_size)
        def closure(data):
            # data[0] rappresenta i sensori
            outputs = model(data[1], data[0])
            optimizer.zero_grad()
            loss = loss_fun(outputs, data[2])
            loss.backward()
            return loss
    else:
        train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
        def closure(data):
            outputs = model(data[0])
            optimizer.zero_grad()
            loss = loss_fun(outputs, data[1])
            loss.backward()
            return loss

    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, epochs + 1):

        # print("epoch ", epoch)
               
        for k, data in enumerate(train_loader):
            model.train()
            # def closure():
            #     outputs = model(data[0], sensors_coords)
            #     optimizer.zero_grad()
            #     loss = loss_fun(outputs, data[1])
            #     loss.backward()
            #     return loss
            loss = closure(data)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_error = loss_output(train_dataset.Y, model(train_dataset.X, sensors_coords_train))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X, sensors_coords_valid))
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)

        if verbose == True:
            print("Epoch "+ str(epoch) + ": Training loss = " + formatter(train_error_list[-1]) + " \t Validation loss = " + formatter(valid_error_list[-1]), flush = True)

        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            model.load_state_dict(best_params)
            train_error = loss_output(train_dataset.Y, model(train_dataset.X, sensors_coords_train))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X, sensors_coords_valid))
            
            if verbose == True:
                print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error))
         
            return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
    
    model.load_state_dict(best_params)
    train_error = loss_output(train_dataset.Y, model(train_dataset.X, sensors_coords_train))
    valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X, sensors_coords_valid))
    
    if verbose == True:
        print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error))
        
    return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()

# =====================================================
# 1. Modulo per il Positional Encoding basato su Fourier
# =====================================================

def fourier_encode(x, B):
    """
    Applica il positional encoding Fourier.
    
    x: tensor di shape (n, d)
    B: tensor di shape (d, D), frequenze da usare per la codifica.
    
    Restituisce un tensore di shape (n, 2*D)
    """
    x_proj = 2 * torch.pi * x @ B  # shape: (n, D)
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SensorDataInterpolator(torch.nn.Module):
    def __init__(self, XY, Vxtrain, B):
        """
        XY: torch.Tensor di shape (ntraj, nvelocity, 2) -> coordinate dei nodi per traiettoria
        Vxtrain: torch.Tensor di shape (ntraj, ntimes, nvelocity) -> valori sui nodi (statici nel tempo)
        B: torch.Tensor di shape (2, D) -> matrice di frequenze per il Fourier encoding
        """
        super().__init__()
        self.XY = XY  # assume che il dato non cambi
        self.Vxtrain = Vxtrain  # il valore da interpolare (rimanente costante)
        self.B = B  # può essere fissata oppure resa learnable (se necessario)
        self.ntraj, self.nvelocity, _ = XY.shape
        self.ntimes = Vxtrain.shape[1]
        # Precompute il positional encoding dei nodi, che non cambierà:
        XY_flat = XY.view(-1, XY.shape[-1])
        self.node_encodings = fourier_encode(XY_flat, self.B).view(self.ntraj, self.nvelocity, -1)
    
    def forward(self, sensors_coords):
        """
        sensors_coords: torch.Tensor di shape (nsensors, 2) (trainable)
        
        Restituisce:
           sensor_data: tensor di shape (ntraj, ntimes, nsensors) ottenuto come interpolazione differenziabile
           dei valori Vxtrain nei nuovi punti dati dai sensors_coords.
        """
        nsensors = sensors_coords.shape[0]
        # Calcola l'encoding dei sensori
        sensor_encodings = fourier_encode(sensors_coords, self.B)  # shape: (nsensors, 2*D)
        # node_encodings: (ntraj, nvelocity, 2*D); vogliamo il prodotto scalare tra
        # ogni sensore ed ogni nodo per ogni traiettoria.
        # Risulta una similarità di shape: (ntraj, nsensors, nvelocity)
        sensor_encodings_expanded = sensor_encodings.unsqueeze(0)  # (1, nsensors, 2*D)
        similarity = torch.matmul(sensor_encodings_expanded, self.node_encodings.transpose(1,2))
        
        # Softmax lungo l'asse dei nodi per ottenere i pesi
        weights = F.softmax(similarity, dim=2)  # shape: (ntraj, nsensors, nvelocity)
        # Per eseguire la somma pesata sui valori:
        # Vxtrain: (ntraj, ntimes, nvelocity)
        # Per la moltiplicazione batch, bisogna trasporre i pesi: (ntraj, nvelocity, nsensors)
        weights_t = weights.transpose(1,2)  # (ntraj, nvelocity, nsensors)
        # Per ogni traiettoria ed ogni timestep: prodotto (ntimes, nvelocity) @ (nvelocity, nsensors)
        # Lo facciamo in batch:
        Vx_flat = self.Vxtrain.view(-1, self.nvelocity)  # (ntraj*ntimes, nvelocity)
        # Ripetiamo i pesi per ogni timestep: weights_t è (ntraj, nvelocity, nsensors)
        weights_expanded = weights_t.repeat_interleave(self.ntimes, dim=0)  # (ntraj*ntimes, nvelocity, nsensors)
        # Riformattiamo Vxtrain e moltiplichiamo:
        Vx_flat = Vx_flat.unsqueeze(1)  # (ntraj*ntimes, 1, nvelocity)
        sensor_vals_flat = torch.bmm(Vx_flat, weights_expanded)  # (ntraj*ntimes, 1, nsensors)
        sensor_vals = sensor_vals_flat.squeeze(1).view(self.ntraj, self.ntimes, nsensors)
        
        return sensor_vals
    
def fit_extended(model, 
                 sensor_interpolator, 
                 train_dataset_out, # già con Padding
                #  valid_dataset_out, 
                 lag, 
                 sensors_coords, 
                 update_interval=10, batch_size=64, epochs=1000, optim_cls=torch.optim.Adam, lr=1e-3, 
                #  loss_fun=torch.nn.MSELoss(),
                 loss_fun = mse, loss_output = mre, formatter = num2p, 
                 verbose=True, patience=50, device='cpu'):
    """
    Training esteso che, ogni update_interval step o epoch, rigenera il dataset relativo agli input sensoriali.
    Il dataset "output" (train_dataset_out) resta invariato.
    
    - model: modello SHRED (o SHREDTransformer) che prende in input una time series dei sensori.
    - sensor_interpolator: modulo che genera la time series dei sensori data la posizione attuale (trainabile).
    - train_dataset_out: output target (invariato), ad es. dati Vxtrain interpolati o altri output.
    - sensors_coords: tensore (trainabile) contenente le posizioni attuali dei sensori.
    
    L'ottimizzazione può aggiornare sia i parametri del modello che le posizioni dei sensori.
    """
    # Preinizializza l'ottimizzatore con i parametri del modello e anche sensors_coords
    optimizer = optim_cls(list(model.parameters()) + [sensors_coords], lr=lr)
    
    # Costruzione iniziale del dataset di input tramite il sensor_interpolator
    sensors_data = sensor_interpolator(sensors_coords)  # shape (ntraj, ntimes, nsensors)
    train_dataset_in = Padding(sensors_data, lag).to(device)
    train_dataset = TimeSeriesDataset(train_dataset_in, train_dataset_out)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # best_valid_loss = float('inf')
    train_error_list = []
    # patience_counter = 0

    for epoch in range(1, epochs + 1):
        
        # Aggiorna periodicamente il dataset delle posizioni sensoriali
        if epoch % update_interval == 0:
            sensors_data = sensor_interpolator(sensors_coords)
            train_dataset = TimeSeriesDataset(Padding(sensors_data, lag).to(device), train_dataset_out)
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
            if verbose:
                print(f"Epoch {epoch}: Aggiornato il dataset con le nuove posizioni dei sensori.")
        
        model.train()
        for k, (X_batch, Y_batch) in enumerate(train_loader):
            print(f"batch {k}")
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            # print(outputs.shape)
            # print(Y_batch.shape)
            loss = loss_fun(outputs, Y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_error = loss_output(train_dataset.Y, model(train_dataset.X))
            # valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            train_error_list.append(train_error)
            # valid_error_list.append(valid_error)

        if verbose == True:
            print("Epoch "+ str(epoch) + ": Training loss = " + formatter(train_error_list[-1])) # + " \t Validation loss = " + formatter(valid_error_list[-1]), flush = True)

        # parte sulla patience da finire
        # if valid_error == torch.min(torch.tensor(valid_error_list)):
        #     patience_counter = 0
        #     best_params = deepcopy(model.state_dict())
        # else:
        #     patience_counter += 1

        # if patience_counter == patience:
        #     model.load_state_dict(best_params)
        #     train_error = loss_output(train_dataset.Y, model(train_dataset.X))
        #     valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            
        #     if verbose == True:
        #         print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error))
         
        #     return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
    
    # model.load_state_dict(best_params)
    # train_error = loss_output(train_dataset.Y, model(train_dataset.X))
    # valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
    
    if verbose == True:
        print("Training done: Training loss = " + formatter(train_error)) #  + " \t Validation loss = " + formatter(valid_error))
        
    return torch.tensor(train_error_list).detach().cpu().numpy() # , torch.tensor(valid_error_list).detach().cpu().numpy()

def forecast(forecaster, input_data, steps, nsensors):
    '''
    Forecast time series in time
    Inputs
    	forecaster model (`torch.nn.Module`)
        starting time series of dimension (ntrajectories, lag, nsensors+nparams)
    	number of forecasting steps
        number of sensors
    Outputs
        forecast of the time series in time
    '''   

    forecast = []
    for i in range(steps):
        forecast.append(forecaster(input_data))
        temp = input_data.clone()
        input_data[:,:-1] = temp[:,1:]
        input_data[:,-1, :nsensors] = forecast[i]

    return torch.stack(forecast, 1)
















