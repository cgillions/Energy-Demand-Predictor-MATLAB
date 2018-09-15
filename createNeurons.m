% Creates 'count' number of neurons with the given width, using values
% from clusterData for their centers and random weights
function [neurons] = createNeurons(count, width, clusterData)
    [~, clusters] = kmeans(clusterData, count, 'distance', 'sqEuclidean', 'start', 'sample');

    % Create a neuron for each cluster, preallocating the array.
    neurons(count, 1) = Neuron();
    for i = 1 : size(clusters, 1)
        neurons(i) = Neuron(clusters(i, :), randomInRange(0, 1), width);
    end
end