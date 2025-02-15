import torch
#C'est la bibliothèque principale de PyTorch, utilisée pour créer et manipuler des tenseurs, ainsi que pour définir des modèles de deep learning.
import torch.nn as nn
#Cette sous-bibliothèque contient des classes et des fonctions pour construire et entraîner des réseaux de neurones, telles que nn.Module, nn.LSTM, nn.Linear, etc.


class LSTMClassifier(nn.Module):
    # Il s'agit de la classe qui définit votre modèle de classification basé sur un LSTM. Elle hérite de torch.nn.Module, ce qui permet de bénéficier des fonctionnalités intégrées de PyTorch pour la gestion du modèle, des paramètres, de l'entraînement, etc.
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        #input_size : La taille de l'entrée
        #hidden_size : La taille de l'état caché dans le LSTM, qui détermine la capacité du modèle à capturer les relations temporelles.
        #num_layers : Le nombre de couches LSTM empilées.
        #num_classes : Le nombre de classes dans le problème de classification.
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #Crée une couche LSTM avec les paramètres spécifiés
        #input_size : Le nombre de caractéristiques par frame
        #hidden_size : Le nombre d'unités dans l'état caché.
        #num_layers : Le nombre de couches LSTM empilées.
        #batch_first=True : Indique que l'entrée et la sortie de la couche LSTM doivent avoir la forme (batch_size, seq_len, input_size) plutôt que (seq_len, batch_size, input_size).
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        #La première couche
        self.fc1 = nn.Sequential(
            #nn.Linear(hidden_size, 1024) : Unité linéaire qui transforme l'entrée de taille hidden_size (c'est-à-dire la sortie du LSTM) en une sortie de taille 1024.
            nn.Linear(hidden_size, 1024),
            #La fonction d'activation ReLU (Rectified Linear Unit) qui ajoute de la non-linéarité au modèle.
            nn.ReLU()
        )

        #La deuxième couche
        self.fc2 = nn.Sequential(
            #Transformation linéaire pour passer de 1024 dimensions à num_classes dimensions.
            nn.Linear(1024, num_classes),
            #La fonction d'activation Softmax qui applique la normalisation exponentielle pour obtenir des probabilités entre 0 et 1 pour chaque classe. Le paramètre dim=1 indique que la normalisation est effectuée sur les colonnes (chaque échantillon dans un lot).
            nn.Softmax(dim=1)
        )

#ce produit automatiquement
    def forward(self, x):
        #La méthode qui définit la propagation avant du modèle
        #L'état caché initial
        #num_layers est le nombre de couches LSTM,
        #batch_size est la taille du lot d'entrées (données d'entrée),
        #hidden_size est la taille de l'état caché.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state
        # L'état de la cellule initiale du LSTM
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state

        #out est la séquence des sorties du LSTM
        out, _ = self.lstm(x, (h0, c0))

        #Utilise uniquement la sortie du dernier pas de temps (la dernière colonne de la sortie du LSTM
        # Use only the output from the last time step
        out = out[:, -1, :]

        #Applique la première couche entièrement connectée (fc1) à la sortie du LSTM. Cela transforme la sortie du LSTM en un vecteur de taille 1024.
        out = self.fc1(out)
        out = self.fc2(out)
        return out
