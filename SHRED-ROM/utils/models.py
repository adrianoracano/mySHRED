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

        self.input_size = input_size
        # # if input_size % nfeat != 0:
        # #     raise RuntimeError("Inconsistent features number")
        self.mix = torch.nn.Linear(nfeat, mix_dim)
        self.lstm = torch.nn.LSTM(input_size = nfeat * mix_dim,
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

        B, T, nsensors, nfeat = x.size()
        # # 1) reshape to isolate per-sensor features
        # x = x.view(B, T, -1, self.mix.in_features)  # -> [B, T, nsensors, nfeat]

        # 2) apply mixing MLP to each sensor's feature vector
        #    will broadcast over B, T, nsensors
        x = self.mix(x)  # -> [B, T, nsensors, mix_dim]

        # 3) flatten sensors back into feature axis
        x = x.view(B, T, -1)  # -> [B, T, nsensors * mix_dim]
        
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

import torch
import torch.nn as nn

# class SHREDagnostic(nn.Module):
#     def __init__(self,
#                  coord_dim,
#                  latent_dim,
#                 #  output_size, 
#                  hidden_size=64,
#                  lstm_layers=2,
#                  decoder_sizes=None,
#                  dropout=0.0):
#         """
#         SHRED model with sensor-agnostic, coordinate-based encoding and decoding.

#         Args:
#             coord_dim (int): dimension of sensor and query coordinates (e.g., 2 for 2D).
#             latent_dim (int): dimension of the global latent representation.
#             hidden_size (int): hidden size for per-sensor LSTM and coordinate MLP.
#             lstm_layers (int): number of layers in per-sensor LSTM.
#             decoder_sizes (list of int): sizes of hidden layers in the decoder MLP.
#             dropout (float): dropout probability between decoder layers.
#         """
#         super(SHREDagnostic, self).__init__()
        
#         # Per-sensor temporal encoder (one-dimensional time series)
#         self.per_sensor_lstm = nn.LSTM(input_size=1,
#                                        hidden_size=hidden_size,
#                                        num_layers=lstm_layers,
#                                        batch_first=True)
        
#         # MLP to embed sensor coordinates
#         self.coord_mlp = nn.Sequential(
#             nn.Linear(coord_dim, hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(inplace=True)
#         )
        
#         # MLP to combine per-sensor temporal embedding and coord embedding
#         self.local_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_size, latent_dim),
#             nn.ReLU(inplace=True)
#         )
        
#         # Decoder MLP: inputs = [global_latent, query_coord]
#         if decoder_sizes is None:
#             decoder_sizes = [latent_dim]
#         layers = []
#         in_dim = latent_dim + coord_dim
#         for out_dim in decoder_sizes:
#             layers.append(nn.Linear(in_dim, out_dim))
#             layers.append(nn.ReLU(inplace=True))
#             if dropout > 0:
#                 layers.append(nn.Dropout(dropout))
#             in_dim = out_dim
#         # Final output layer to scalar prediction per query point
#         layers.append(nn.Linear(in_dim, 1))
#         self.decoder_mlp = nn.Sequential(*layers)

#     def forward(self, sensor_hist, sensor_coords, query_coords):
#         """
#         Forward pass.

#         Args:
#             sensor_hist (torch.Tensor): shape (B, N_s, L), time series of sensor values.
#             sensor_coords (torch.Tensor): shape (B, N_s, coord_dim), coordinates of sensors.
#             query_coords (torch.Tensor): shape (B, N_q, coord_dim), points where to reconstruct the field.

#         Returns:
#             torch.Tensor: shape (B, N_q), reconstructed field values at query_coords.
#         """
#         B, N_s, L = sensor_hist.size()
#         # Encode temporal history per sensor
#         # reshape to (B*N_s, L, 1) for LSTM
#         hist = sensor_hist.view(B * N_s, L, 1)
#         print("ok1")
#         _, (h_n, _) = self.per_sensor_lstm(hist)
#         # take last layer hidden state, shape (B*N_s, hidden_size)
#         print("ok2")
#         h_seq = h_n[-1].view(B, N_s, -1)
#         print("ok3")
#         # Embed sensor coordinates
#         coord_emb = self.coord_mlp(sensor_coords.view(B * N_s, -1))
#         coord_emb = coord_emb.view(B, N_s, -1)
#         print("ok4")
#         # Local embedding per sensor
#         local_in = torch.cat([h_seq, coord_emb], dim=-1)  # (B, N_s, 2*hidden_size)
#         local_emb = self.local_mlp(local_in)               # (B, N_s, latent_dim)
#         print("ok5")
#         # Aggregate over sensors (mean pooling)
#         global_h = local_emb.mean(dim=1)                   # (B, latent_dim)
        
#         # Decode for each query point
#         B, N_q, _ = query_coords.size()
#         # expand global latent to each query
#         h_exp = global_h.unsqueeze(1).expand(-1, N_q, -1)  # (B, N_q, latent_dim)
#         dec_in = torch.cat([h_exp, query_coords], dim=-1)  # (B, N_q, latent_dim + coord_dim)
#         u_pred = self.decoder_mlp(dec_in)   
#         print("ok6")               # (B, N_q, 1)
#         return u_pred.squeeze(-1)  # (B, N_q)
    
    
#     def freeze(self):

#         self.eval()
        
#         for param in self.parameters():
#             param.requires_grad = False

#     def unfreeze(self):

#         self.train()
        
#         for param in self.parameters():
#             param.requires_grad = True

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
        # Coordinate-based decoder (if used)
        if decoder_sizes is None:
            coord_dec = [latent_dim]
        else:
            coord_dec = decoder_sizes
        layers = []
        in_dim = latent_dim + coord_dim
        for out_dim in coord_dec:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.coord_decoder = nn.Sequential(*layers)

    def forward(self, sensor_hist, sensor_coords, query_coords=None):
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
        if query_coords is None:
            return self.coef_decoder(global_h)  # (B, output_size)
        # Coordinate-based field reconstruction
        B, N_q, _ = query_coords.size()
        h_exp = global_h.unsqueeze(1).expand(-1, N_q, -1)
        dec_in = torch.cat([h_exp, query_coords], dim=-1)
        u_pred = self.coord_decoder(dec_in)
        return u_pred.squeeze(-1)  # (B, N_q)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return x

class SHREDTransformer(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=2, 
                 decoder_sizes=[350, 400], dropout=0.0, nhead=4, d = 2):
        '''
        SHREDTransformer model definition
         
        Inputs:
            input_size: dimensione dell'input (es. numero di sensori)
            output_size: dimensione dell'output (es. dimensione della variabile full-order)
            hidden_size: dimensione dei layer interni (default=64)
            hidden_layers: numero dei layer dell’encoder Transformer (default=2)
            decoder_sizes: lista delle dimensioni dei layer del decoder (default=[350, 400])
            dropout: parametro di dropout (default=0.0)
            nhead: numero di teste di attenzione per il Transformer (default=4)
            d: dimensione dello spazio (default=2 (2D))
        '''
        super(SHREDTransformer, self).__init__()
        
        # Proiezione dell'input da input_size a hidden_size, dato che il Transformer lavora con d_model = hidden_size.
        self.input_projection = torch.nn.Linear(input_size, hidden_size)

        # Proiezione della posizione di un sensore da 2 (x, y) a hidden_size
        self.sensor_coord_proj = torch.nn.Linear(2, hidden_size)

        # Aggiunta del positional encoding per la sequenza
        self.positional_encoding = PositionalEncoding(d_model=hidden_size)
        
        # Definizione di un layer base del TransformerEncoder.
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)

        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=hidden_layers)
        
        # Costruzione del decoder come sequenza di layer lineari (e opzionalmente dropout e ReLU).
        self.decoder = torch.nn.ModuleList()
        # Inseriamo la dimensione iniziale derivante dal Transformer
        decoder_sizes.insert(0, hidden_size)
        decoder_sizes.append(output_size)
        for i in range(len(decoder_sizes) - 1):
            self.decoder.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            # Aggiungiamo dropout e ReLU per tutti i layer tranne l'ultimo
            if i != len(decoder_sizes) - 2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())
                
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    # def forward(self, x, sensors_coords):

    #     # sensors_coords ha shape (nsensors, d)
    #     coord_emb = self.sensor_coord_proj(sensors_coords)   # (input_size, hidden_size)

    #     # x ha shape: (batch_size, seq_len, input_size)
    #     # Proiezione dell'input
    #     print(x.shape)
    #     print(coord_emb.shape)
    #     x = self.input_projection(x)  # -> (batch_size, seq_len, hidden_size)
    #     print(x.shape)
        
    #     # Il Transformer richiede l'input in shape (seq_len, batch_size, hidden_size)
    #     x = x.transpose(0, 1)
    #     print(x.shape)

    #     x = self.positional_encoding(x)
    #     print(x.shape)
        
    #     # Passaggio attraverso l'encoder Transformer
    #     x = self.transformer_encoder(x) # + coord_emb.unsqueeze(0).unsqueeze(0) # -> (seq_len, batch_size, hidden_size)
    #     print(x.shape)

    #     x += coord_emb.unsqueeze(0).unsqueeze(0)
    #     # Possiamo scegliere di prendere l'ultimo output della sequenza come rappresentazione
    #     output = x[-1]  # -> (batch_size, hidden_size)
        
    #     # Passaggio attraverso il decoder
    #     for layer in self.decoder:
    #         output = layer(output)
            
    #     return output

    def forward(self, x, sensors_coords):
        """
        x: Tensor di shape (batch_size, seq_len, nsensors)
        sensors_coords: Tensor di shape (batch_size, nsensors, coord_dim)
        """

        # 1) Calcolo embedding spaziale per ogni batch
        #    coord_emb: (batch_size, nsensors, hidden_size)
        coord_emb = self.sensor_coord_proj(sensors_coords)

        # 2) Proiezione "classica" dei segnali sensoriali
        #    x_proj: (batch_size, seq_len, hidden_size)
        x_proj = self.input_projection(x)

        # 3) Costruisco la componente spaziale pesata
        #    x.unsqueeze(-1):            (B, T, S, 1)
        #    coord_emb.unsqueeze(1):     (B, 1, S, H)
        #    -> prodotto broadcast:      (B, T, S, H)
        #    sommo sulla dimsensori S →  (B, T, H)
        x_spatial = (x.unsqueeze(-1) * coord_emb.unsqueeze(1)).sum(dim=2)

        # 4) Unisco la componente spaziale a quella "classica"
        #    x_comb: (batch_size, seq_len, hidden_size)
        x_comb = x_proj + x_spatial

        # 5) Trasposizione per Transformer: (seq_len, batch_size, hidden_size)
        x_trans = x_comb.transpose(0, 1)

        # 6) Aggiungo il positional encoding temporale
        x_pe = self.positional_encoding(x_trans)

        # 7) Passaggio nell’encoder Transformer
        out_enc = self.transformer_encoder(x_pe)  # (seq_len, batch_size, hidden_size)

        # 8) Estraggo l’ultimo time‐step come rappresentazione
        rep = out_enc[-1]  # (batch_size, hidden_size)

        # 9) Decoder finale
        output = rep
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

class SpatialPreEncoder(torch.nn.Module):
    def __init__(self, coord_dim, d_model, nhead):
        super().__init__()
        # project coordinates to feature space
        self.coord_proj = torch.nn.Linear(coord_dim, d_model)
        # self-attention among sensors
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, sensors_coords):
        # sensors_coords: (S, coord_dim)
        S = sensors_coords.size(0)
        # 1) project coords -> (S, d_model)
        coord_emb = self.coord_proj(sensors_coords)
        # 2) add batch dim -> (1, S, d_model)
        coord_seq = coord_emb.unsqueeze(0)
        # 3) self-attention: query=key=value
        attn_out, _ = self.self_attn(coord_seq, coord_seq, coord_seq)
        # 4) remove batch dim -> (S, d_model)
        return attn_out.squeeze(0)

class SHREDPerceiverSpatial(torch.nn.Module):
    def __init__(self, nsensors, coord_dim, 
                 output_size, hidden_size=64, hidden_layers=2,
                 decoder_sizes=[350, 400], dropout=0.0, d_model = 64, attn_heads=4):
        super().__init__()
        # 1) spatial encoder for sensor positions
        self.spatial_encoder = SpatialPreEncoder(coord_dim, d_model, attn_heads)
        # 2) reading embedder: map scalar reading to d_model
        self.reading_proj = torch.nn.Linear(1, d_model)
        # 3) LSTM over time: input_size = nsensors * d_model
        self.input_size = nsensors * d_model
        self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                  hidden_size=hidden_size,
                                  num_layers=hidden_layers,
                                  batch_first=True)
        # 4) decoder MLP
        layers = [hidden_size] + decoder_sizes + [output_size]
        self.decoder = torch.nn.ModuleList()
        for i in range(len(layers)-1):
            self.decoder.append(torch.nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.nsensors = nsensors
        self.d_model = d_model

    def forward(self, x, sensors_coords):
        """
        x: Tensor of shape (B, T, S) -- sensor readings
        sensors_coords: Tensor of shape (S, coord_dim)
        """
        B, T, S = x.shape
        # 1) encode spatial positions -> (S, d_model)
        coord_emb = self.spatial_encoder(sensors_coords)
        # 2) project readings -> (B, T, S, d_model)
        x_unsq = x.unsqueeze(-1)              # B, T, S, 1
        reading_emb = self.reading_proj(x_unsq)  # B, T, S, d_model
        # 3) fuse: sum reading and spatial
        coord_exp = coord_emb.unsqueeze(0).unsqueeze(0).expand(B, T, S, self.d_model)
        fused = reading_emb + coord_exp        # B, T, S, d_model
        # 4) flatten sensors: (B, T, S * d_model)
        fused_flat = fused.view(B, T, S * self.d_model)
        # 5) LSTM temporal
        h0 = torch.zeros(self.hidden_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros_like(h0)
        if next(self.parameters()).is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        _, (hn, _) = self.lstm(fused_flat, (h0, c0))
        rep = hn[-1]                           # B, hidden_size
        # 6) decoder
        out = rep
        for layer in self.decoder:
            out = layer(out)
        return out
    
    def freeze(self):

        self.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):

        self.train()
        
        for param in self.parameters():
            param.requires_grad = True

class SHREDConcat(torch.nn.Module):
    def __init__(self, nsensors, coord_pe_dim, output_size,
                 hidden_size=64, hidden_layers=2,
                 decoder_sizes=[350, 400], dropout=0.0):
        '''
        SHRED with simple concatenation of sensor positional encodings

        nsensors: number of sensors (S)
        coord_pe_dim: dimensionality of each sensor's positional encoding (2*D)
        output_size: dimension of model output
        hidden_size, hidden_layers, decoder_sizes, dropout: as in base SHRED
        '''
        super().__init__()
        # New input feature size: per time-step, we concatenate sensor readings and PE
        # Flattened feature size = S (readings) + S*coord_pe_dim
        self.input_size = nsensors + nsensors * coord_pe_dim
        # LSTM
        self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                  hidden_size=hidden_size,
                                  num_layers=hidden_layers,
                                  batch_first=True)
        # Decoder
        self.decoder = torch.nn.ModuleList()
        layers = [hidden_size] + decoder_sizes + [output_size]
        for i in range(len(layers) - 1):
            self.decoder.append(torch.nn.Linear(layers[i], layers[i+1]))
            if i != len(layers) - 2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.nsensors = nsensors
        self.coord_pe_dim = coord_pe_dim

    def forward(self, x, sensors_pe):
        '''
        x:           (B, T, S)  sensor readings
        sensors_pe:  (S, coord_pe_dim) positional encodings per sensor
        '''
        B, T, S = x.shape
        # 1) Flatten sensor readings per time-step: x_flat = (B, T, S)
        # 2) Flatten positional encodings: pe_flat = (S * coord_pe_dim,)
        pe_flat = sensors_pe.view(-1)  # (S*coord_pe_dim)
        # 3) Expand pe_flat to (B, T, S*coord_pe_dim)
        pe_exp = pe_flat.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        # 4) Concatenate along feature dimension: (B, T, S + S*coord_pe_dim)
        x_cat = torch.cat([x, pe_exp], dim=2)
        # 5) LSTM
        # init hidden states
        h0 = torch.zeros(self.hidden_layers, B, self.hidden_size,
                         device=x.device)
        c0 = torch.zeros_like(h0)
        if next(self.parameters()).is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        lstm_out, (hn, _) = self.lstm(x_cat, (h0, c0))
        rep = hn[-1]  # (B, H)
        # 6) Decoder
        out = rep
        for layer in self.decoder:
            out = layer(out)
        return out
    
    def freeze(self):

        self.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):

        self.train()
        
        for param in self.parameters():
            param.requires_grad = True

class SHREDAttention(torch.nn.Module):

    def __init__(self, input_size, output_size,
                 hidden_size=64, hidden_layers=2,
                 decoder_sizes=[350, 400], dropout=0.0,
                 coord_dim=2, attn_heads=4):
        super().__init__()
        # --- spatial attention setup ---
        # Proietta valore di ogni sensore
        self.sensor_proj = torch.nn.Linear(1, hidden_size)
        # Proietta coordinate sensori in embedding
        self.coord_proj = torch.nn.Linear(coord_dim, hidden_size)
        # Attention fra sensori (seq_len = nsensors)
        self.spatial_att = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            batch_first=True
        )
        # --- LSTM temporale ---
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True
        )
        # --- Decoder ---
        self.decoder = torch.nn.ModuleList()
        sizes = [hidden_size] + decoder_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.decoder.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            if i != len(sizes) - 2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.coord_dim = coord_dim

    def forward(self, x, sensors_coords):
        """
        x: Tensor B x T x S
        sensors_coords: Tensor S x coord_dim  oppure  B x S x coord_dim
        """
        B, T, S = x.shape
        H = self.hidden_size

        # 1) Embed each sensor reading
        x_unsq = x.unsqueeze(-1)               # B x T x S x 1
        x_emb = self.sensor_proj(x_unsq)        # B x T x S x H
        x_flat = x_emb.reshape(B * T, S, H)     # (B*T) x S x H

        # 2) Embed sensor coordinates
        # handle shared vs per-batch coords
        if sensors_coords.dim() == 2:
            # S x coord_dim -> S x H
            coord_emb = self.coord_proj(sensors_coords)  # S x H
            coord_emb = coord_emb.unsqueeze(0).unsqueeze(0).expand(B, T, S, H)
        else:
            # B x S x coord_dim -> B x S x H
            coord_emb = self.coord_proj(sensors_coords)  # B x S x H
            coord_emb = coord_emb.unsqueeze(1).expand(B, T, S, H)
        coord_flat = coord_emb.reshape(B * T, S, H)

        # 3) Spatial attention with coordinate bias
        q = x_flat
        k = x_flat + coord_flat
        v = x_flat + coord_flat
        attn_out, _ = self.spatial_att(q, k, v)  # (B*T) x S x H

        # 4) Pool across sensors -> sequence features
        x_spatial = attn_out.mean(dim=1)        # (B*T) x H
        x_seq = x_spatial.view(B, T, H)        # B x T x H

        # 5) Temporal LSTM
        h0 = torch.zeros(self.hidden_layers, B, H, device=x.device)
        c0 = torch.zeros_like(h0)
        seq_out, (hn, _) = self.lstm(x_seq, (h0, c0))
        rep = hn[-1]                             # B x H

        # 6) Decoder finale
        out = rep
        for layer in self.decoder:
            out = layer(out)
        return out

    def freeze(self):

        self.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):

        self.train()
        
        for param in self.parameters():
            param.requires_grad = True

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
            print(f"batch {k}")
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

def fit_with_memory_monitoring(model, train_dataset, valid_dataset, batch_size=64, epochs=1000,
                               optim=torch.optim.Adam, lr=1e-3, loss_fun=mse, loss_output=mre,
                               formatter=num2p, verbose=True, patience=100, monitor_interval=10, device="cuda"):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    optimizer = optim(model.parameters(), lr=lr)

    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = deepcopy(model.state_dict())
    global_step = 0

    for epoch in range(1, epochs + 1):

        for k, data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            outputs = model(data[0].to(device))
            loss = loss_fun(outputs, data[1].to(device))
            loss.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        with torch.no_grad():
            train_error = loss_output(train_dataset.Y.to(device), model(train_dataset.X.to(device)))
            valid_error = loss_output(valid_dataset.Y.to(device), model(valid_dataset.X.to(device)))
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)

        if verbose:
            print("Epoch {:d}: Training loss = {} \t Validation loss = {}".format(
                epoch, formatter(train_error_list[-1]), formatter(valid_error_list[-1])
            ))

            # Monitoraggio della memoria ogni 'monitor_interval' epoche
            if epoch % monitor_interval == 0:
                mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
                print("GPU memory allocated: {:.2f} MB, reserved: {:.2f} MB".format(mem_alloc, mem_reserved))
                # Puoi anche stampare un report più completo se vuoi:
                # print(torch.cuda.memory_summary(device=device))

        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            model.load_state_dict(best_params)
            train_error = loss_output(train_dataset.Y.to(device), model(train_dataset.X.to(device)))
            valid_error = loss_output(valid_dataset.Y.to(device), model(valid_dataset.X.to(device)))
            if verbose:
                print("Training done: Training loss = {} \t Validation loss = {}".format(
                    formatter(train_error), formatter(valid_error)
                ))
            return (torch.tensor(train_error_list).detach().cpu().numpy(),
                    torch.tensor(valid_error_list).detach().cpu().numpy())

    model.load_state_dict(best_params)
    train_error = loss_output(train_dataset.Y.to(device), model(train_dataset.X.to(device)))
    valid_error = loss_output(valid_dataset.Y.to(device), model(valid_dataset.X.to(device)))
    if verbose:
        print("Training done: Training loss = {} \t Validation loss = {}".format(
            formatter(train_error), formatter(valid_error)
        ))
    return (torch.tensor(train_error_list).detach().cpu().numpy(),
            torch.tensor(valid_error_list).detach().cpu().numpy())

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
















