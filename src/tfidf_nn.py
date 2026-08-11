import torch
import torch.nn as nn



class TFIDFNeuralNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes=5
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),


            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),


            nn.Linear(
                64,
                num_classes
            )
        )



    def forward(self,x):

        return self.network(x)
