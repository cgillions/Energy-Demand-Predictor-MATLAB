% Represents a node within the network.
classdef Neuron
   properties
       % Center values for each dimension of the neuron.
       centers
       
       % The output weight of the node.
       weight;
       
       % The width of the Gaussian function.
       width;
   end
   
   methods
       % Constructor for a Neuron in the network.
       function neuron = Neuron(centers, weight, width)
           if nargin == 0
              neuron.centers = [0 0 0];
              neuron.weight  = 0;
              neuron.width   = 0;
           else
              neuron.centers = centers;
              neuron.weight = weight;
              neuron.width  = width;
           end
       end
       
       % Method to calculate the activation value of the neuron for a given
       % input.
       function activationValue = activation(neuron, inputs)
           diffs       = neuron.centers - inputs;
           squareDiffs = diffs .^ 2;
           
           activationValue = 1;
           for i = 1:size(squareDiffs, 2)
               activationValue = activationValue * exp(-squareDiffs(i) / (2 * neuron.width ^ 2));
           end
       end
   end
end