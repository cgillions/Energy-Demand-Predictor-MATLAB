classdef Network < handle
    properties
        % The vector of neurons belonging to the network.
        neurons;
        
        % The learning rate of the network.
        learningRate;
    end
    
    methods
        % Constructor for the network.
        function network = Network(learningRate, neuronCount, neuronWidth, clusterData)
            if (nargin ~= 0)               
                % Create the neurons.
                network.learningRate = learningRate;
                network.neurons      = createNeurons(neuronCount, neuronWidth, clusterData);
            end
        end
        
        % Feeds the input value to the single layer network.
        function [output, activationValues] = feed(network, input)
            % Work out the size of the network.
            neuronCount = size(network.neurons, 1);

            % Initialise matrices to hold the neuron's output values.
            activationValues = zeros(neuronCount, 1);
            outputValues     = zeros(neuronCount, 1);

            % For the given input, store the activation value for each neuron.
            for neuronIndex = 1:neuronCount
                % Get the neuron's activatstrion.
                neuron     = network.neurons(neuronIndex);
                activation = neuron.activation(input);

                % Append the result matrices.
                activationValues(neuronIndex) = activation;
                outputValues(neuronIndex)     = activation * neuron.weight;
            end

            % Now we have the activation value and weighted output value for each 
            % hidden node, we can compute the output of the network.
            output = sum(outputValues) / sum(activationValues);
        end
        
        % Feeds the input vector to the single layer network.
        function [outputVector, activationValues] = feedBatch(network, input)
            % Work out the size of the network & input data.
            neuronCount = size(network.neurons, 1);
            dataCount   = size(input(:, 1), 1); 
            
            % Initialise matrices to hold the neuron's output values.
            activationValues = zeros(dataCount, neuronCount);
            outputValues     = zeros(dataCount, neuronCount);
            
            % For each data point, store the activation value for each neuron.
            for dataIndex = 1 : dataCount
                for neuronIndex = 1 : neuronCount
                    % Get the neuron's activation.
                    neuron     = network.neurons(neuronIndex);
                    activation = neuron.activation(input(dataIndex, :));
                    
                    % Append the result matrices.
                    activationValues(dataIndex, neuronIndex) = activation;
                    outputValues(dataIndex, neuronIndex) = activation * neuron.weight;
                end
            end

            % Now we have the activation values and weighted output values for each
            % data point for each hidden node, we can compute the output of the network.
            outputVector = zeros(dataCount, 1);
            
            for dataIndex = 1 : dataCount
                outputVector(dataIndex) = sum(outputValues(dataIndex, :)) / sum(activationValues(dataIndex, :));
            end
        end
        
        function train(network, input, target)
            % Find the output of the network for the data point.
            if size(input, 1) > 1
                [output, activationValues] = network.feedBatch(input);
                
                % Updates the weight of each neuron based on the input, output & target.
                for dataIndex = 1:size(input(:, 1), 1)
                    for neuronIndex = 1:size(network.neurons, 1)
                        neuron = network.neurons(neuronIndex);
                        neuron.weight = neuron.weight + network.learningRate * (target(dataIndex) - output(dataIndex)) * activationValues(dataIndex, neuronIndex);
                        network.neurons(neuronIndex) = neuron;
                    end
                end
            else
                [output, activationValues] = network.feed(input);
                
                % Updates the weight of each neuron based on the input, output & target.
                for neuronIndex = 1:size(network.neurons, 1)
                    network.neurons(neuronIndex).weight = network.neurons(neuronIndex).weight + network.learningRate * (target - output) * activationValues(neuronIndex);
                end
            end
        end
    end
end