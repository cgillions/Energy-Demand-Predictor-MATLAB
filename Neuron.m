% Represents a node within the network.
classdef Neuron
   properties
       % Center of the activation function's first dimension.
       center1;
       
       % Center of the activation function's second dimension.
       center2;
       
       % Center of the activation function's third dimension.
       center3;
       
       % The output weight of the node.
       weight;
       
       % The width of the Gaussian function.
       width;
   end
   
   methods
       % Constructor for a Neuron in the network.
       function neuron = Neuron(center1, center2, center3, weight, width)
           if nargin == 0
              neuron.center1 = 0;
              neuron.center2 = 0;
              neuron.center3 = 0;
              neuron.weight  = 0;
              neuron.width   = 0;
           else
              neuron.center1 = center1;
              neuron.center2 = center2;
              neuron.center3 = center3;
              neuron.weight = weight;
              neuron.width  = width;
           end
       end
       
       % Method to calculate the activation value of the neuron for a given
       % input.
       function activationValue = activation(neuron, x1, x2, x3)
            euclidianDistance = (((x1 - neuron.center1) ^ 2) + ...
               ((x2 - neuron.center2) ^ 2) + ((x3 - neuron.center3) ^ 2)) / 3;
           
           activationValue = exp(-(euclidianDistance / (2 * neuron.width ^ 2)));
       end
   end
end