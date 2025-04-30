from .models import SHRED, fit
import torch
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy

from .utilities import *
import torch.nn.functional as F


# class 

# def fourier_encode(x, B):
#     """
#     Applica il positional encoding Fourier alle coordinate.

#     Parametri:
#       - x: tensor di shape (n, d) (ad esempio, coordinate spaziali)
#       - B: tensor di shape (d, D) contenente le frequenze.

#     Restituisce:
#       - encoding: tensor di shape (n, 2*D) ottenuto concatenando sin(xB) e cos(xB).
#     """
#     # Proiezione: x @ B produce un tensore di shape (n, D)
#     x_proj = 2 * torch.pi * x @ B
#     return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# def generate_sensor_data(sensors_coords_new, B, node_encodings, ntimes, dataset):

#     sensor_encodings_expanded = fourier_encode(sensors_coords_new, B).unsqueeze(0)  # shape: (1, nsensors, 2*D)

#     # - per train, valid e test:
#     similarity = torch.matmul(sensor_encodings_expanded, node_encodings.transpose(1,2))
#     weights_t = F.softmax(similarity, dim=2).transpose(1,2)  # shape: (ntraj, nvelocity, nsensors)
#     weights_expanded = weights_t.repeat_interleave(ntimes, dim=0)  # shape: (ntraj*ntimes, nvelocity, nsensors)
#     sensors_data_flat = torch.bmm(Vxtrain_reshaped, weights_expanded)  # shape: (ntraj*ntimes, 1, nsensors)
#     sensors_data = Vx_interp_flat.squeeze(1).view(train_trajectories, ntimes, nsensors)


# Classe wrapper che ingloba SHRED e gestisce i parametri trainable delle posizioni dei sensori
class SHREDWrapper(nn.Module):
    def __init__(self, 
                 shred_input_size, 
                 shred_output_size, 
                 sensor_init_positions, 
                 domain_bounds, 
                 obstacle_coords,
                 hidden_size=64,
                 hidden_layers=2,
                 decoder_sizes=[350, 400],
                 dropout=0.0):
        """
        sensor_init_positions: tensor di dimensione (num_sensors, 2) con le posizioni iniziali
        domain_bounds: dizionario con i limiti del dominio, ad es. {'Xmin': a, 'Xmax': b, 'Ymin': c, 'Ymax': d}
        obstacle_coords: dizionario con le coordinate dell'ostacolo, ad es. {'Xmin': a, 'Xmax': b, 'Ymin': c, 'Ymax': d}
        """
        super(SHREDWrapper, self).__init__()
        self.shred = SHRED(shred_input_size, shred_output_size, hidden_size, hidden_layers, decoder_sizes, dropout)
        # Parametro trainable per le posizioni dei sensori
        self.sensor_positions = nn.Parameter(sensor_init_positions)
        self.domain_bounds = domain_bounds
        self.obstacle_coords = obstacle_coords

    def forward(self, full_dataset):
        # Aggiorna le posizioni per farle "rimbalzare" se necessario
        valid_sensor_positions = self.bounce_positions(self.sensor_positions)
        # Interpolazione: calcola l'input fluidodinamico alle posizioni aggiornate
        sensor_inputs = interpolate_data(full_dataset, valid_sensor_positions)
        # Passa i dati interpolati al modello SHRED
        output = self.shred(sensor_inputs)
        return output

    def bounce_positions(self, positions):
        """
        "Rimbalza" le posizioni fuori dominio o all'interno dell'ostacolo
        Le operazioni vengono effettuate in modo differenziabile (usando torch.where).
        """
        # Estrai coordinate
        x = positions[:, 0]
        y = positions[:, 1]
        
        # Limiti del dominio
        Xmin, Xmax = self.domain_bounds['Xmin'], self.domain_bounds['Xmax']
        Ymin, Ymax = self.domain_bounds['Ymin'], self.domain_bounds['Ymax']

        # Se fuori dai limiti, rimbalza la posizione invertendo il "sovrappasso"
        x = torch.where(x < Xmin, 2 * Xmin - x, x)
        x = torch.where(x > Xmax, 2 * Xmax - x, x)
        y = torch.where(y < Ymin, 2 * Ymin - y, y)
        y = torch.where(y > Ymax, 2 * Ymax - y, y)
        
        # Controllo se la posizione è interna all'ostacolo (ipotizziamo un ostacolo rettangolare)
        obs_Xmin, obs_Xmax = self.obstacle_coords['Xmin'], self.obstacle_coords['Xmax']
        obs_Ymin, obs_Ymax = self.obstacle_coords['Ymin'], self.obstacle_coords['Ymax']
        
        # Definisce una maschera booleana per le posizioni all'interno dell'ostacolo
        inside_obstacle = point_in_obstacle(x, self.obstacle_coords)
        
        # Se all'interno, "rimbalza" spostando il sensore verso il bordo più vicino dell'ostacolo
        # Calcola le distanze per capire a quale bordo è più vicino:
        dx_left  = (x - obs_Xmin).abs()
        dx_right = (obs_Xmax - x).abs()
        dy_bottom = (y - obs_Ymin).abs()
        dy_top    = (obs_Ymax - y).abs()
        
        # Sceglie il bordo orizzontale se la distanza è minore rispetto a quello verticale
        x_adjusted = torch.where(
            inside_obstacle & (dx_left < dx_right),
            obs_Xmin,  # sposta sul bordo sinistro
            torch.where(inside_obstacle, obs_Xmax, x)
        )
        y_adjusted = torch.where(
            inside_obstacle & (dy_bottom < dy_top),
            obs_Ymin,  # sposta sul bordo inferiore
            torch.where(inside_obstacle, obs_Ymax, y)
        )
        
        # Combina le coordinate (in un caso reale si potrebbe migliorare la logica per spostare in modo consistente)
        bounced_positions = torch.stack([x_adjusted, y_adjusted], dim=1)
        return bounced_positions

# Esempio di training: la funzione fit dovrà ora gestire anche i parametri relativi alle posizioni dei sensori.
def fit_sensors(model, train_dataset, valid_dataset, batch_size=64, epochs=4000, 
        optim=torch.optim.Adam, lr=1e-3, loss_fun=nn.MSELoss(), 
        loss_output=lambda y_pred, y: torch.mean(torch.abs((y_pred-y)/y)), 
        formatter=lambda x: f"{x:.2%}", verbose=False, patience=5):
    '''
    Training del modello che include i parametri trainable delle posizioni dei sensori.
    Gli input train_dataset e valid_dataset devono avere le proprietà:
        - .X ed .Y, oppure essere gestiti opportunamente nel forward tramite funzioni di interpolazione.
    '''
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    # L'ottimizzatore include i parametri del modello wrapper, che comprendono SHRED e le posizioni dei sensori.
    optimizer = optim(model.parameters(), lr=lr)
    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = deepcopy(model.state_dict())
    
    for epoch in range(1, epochs + 1):
        if verbose:
            print("Epoch", epoch)
        for _, data in enumerate(train_loader):
            model.train()
            # Supponiamo che il batch contenga i dati fluidodinamici completi per l'interpolazione
            def closure():
                optimizer.zero_grad()
                # data potrebbe essere il dataset completo, oppure un batch di informazioni necessarie per l'interpolazione
                outputs = model(data)
                loss = loss_fun(outputs, data.Y)  # adatta l'accesso ai target in base all'organizzazione del batch
                loss.backward()
                return loss
            optimizer.step(closure)
        
        model.eval()
        with torch.no_grad():
            # Calcola gli errori di training e validazione usando l'interpolazione completa
            train_out = model(train_dataset)
            valid_out = model(valid_dataset)
            train_error = loss_output(train_out, train_dataset.Y)
            valid_error = loss_output(valid_out, valid_dataset.Y)
            train_error_list.append(train_error.item() if hasattr(train_error, "item") else train_error)
            valid_error_list.append(valid_error.item() if hasattr(valid_error, "item") else valid_error)

        if verbose:
            print(f"Epoch {epoch}: Training loss = {formatter(train_error_list[-1])} \t Validation loss = {formatter(valid_error_list[-1])}")
        
        if valid_error == min(valid_error_list):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter == patience:
            model.load_state_dict(best_params)
            if verbose:
                print(f"Training terminato per early stopping alla Epoch {epoch}.")
            break

    # Carica i migliori parametri trovati
    model.load_state_dict(best_params)
    return train_error_list, valid_error_list