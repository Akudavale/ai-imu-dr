#work till now
"""
Changes planes in Mesnet to measuve covariance
"""
class MesNet(torch.nn.Module):
        """
        Measurement Network (MesNet) for the Implicit Extended Kalman Filter (IEKF).

        This network takes the input sensor data and predicts the measurement covariance
        for the IEKF. The input data consists of the sensor readings, and the output is
        the predicted measurement covariance.

        Parameters:
            None

        Attributes:
            beta_measurement (torch.Tensor): Scaling factor for the predicted covariance.
            tanh (torch.nn.Tanh): Hyperbolic tangent activation function.

            Convolutional Neural Network (CNN) Layers:
                cov_net (torch.nn.Sequential): CNN layers for processing sensor data.
                cov_lin (torch.nn.Sequential): Linear layer for predicting the covariance.

        Input:
            u (torch.Tensor): Input sensor data tensor with shape (batch_size, num_features=6, sequence_length=6000).
            num_features = 6D pose (acc and gyro in x,y,z direction)
        Output:
            - measurements_covs (torch.Tensor): Predicted measurement covariance tensor with shape (batch_size, 2),
            where 2 represents the two elements of the covariance matrix (variance for gyro and acc).
          """
        def __init__(self):
            super(MesNet, self).__init__()
            self.beta_measurement = 3*torch.ones(2).double()
            self.tanh = torch.nn.Tanh()

            self.cov_net1 = torch.nn.Sequential(
                    torch.nn.Conv1d(6, 32, 5),
                    torch.nn.ReplicationPad1d(4),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.5)
            ).double()
            self.cov_net2 = torch.nn.Sequential(
                    torch.nn.Conv1d(32, 32, 5),
                    torch.nn.ReplicationPad1d(4),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.5)
            ).double()
            self.cov_net3 = torch.nn.Sequential(
                    torch.nn.Conv1d(32, 32, 5, dilation=3),
                    torch.nn.ReplicationPad1d(4),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.5),
                    ).double()
            "CNN for measurement covariance"
            self.cov_lin = torch.nn.Sequential(torch.nn.Linear(32, 2), 
                                              torch.nn.Tanh(),
                                              ).double()
            self.cov_lin[0].bias.data[:] /= 100
            self.cov_lin[0].weight.data[:] /= 100

        def forward(self, u, iekf):
            #print(f"input to network is{u} and shape is {u.shape}")
            y_cov = self.cov_net1(u)
            y_cov = self.cov_net2(y_cov)
            y_cov = self.cov_net3(y_cov).transpose(0, 2).squeeze()
            #print(f"output of conv_net is{y_cov} and shape is {y_cov.shape}")
            z_cov = self.cov_lin(y_cov)
            #print(f"output of conv_lin is{z_cov} and shape is {z_cov.shape}")
            z_cov_net = self.beta_measurement.unsqueeze(0)*z_cov
            #print(f"output of beta mesurments is{z_cov_net} and shape is {z_cov_net.shape}")
            measurements_covs = (iekf.cov0_measurement.unsqueeze(0) * (10**z_cov_net)) 
            """ (10**z_cov_net) apply exponentiation with covariance matrices to ensure positive-definiteness.
                The diagonal elements of a covariance matrix represent the variances of individual random variables.
                Variance is a measure of the spread or dispersion of a random variable. A negative variance would not make sense in this context."""
            #print(f"mesurment cov is{measurements_covs} and shape is {measurements_covs.shape}")
            return measurements_covs
        

lr_mesnet = {'cov_net1': 1e-4,'cov_net2': 1e-4, 'cov_net2': 1e-4,'cov_lin': 1e-4,}
weight_decay_mesnet = {'cov_net1': 1e-4,'cov_net2': 1e-4, 'cov_net2': 1e-4,'cov_lin': 1e-8,}