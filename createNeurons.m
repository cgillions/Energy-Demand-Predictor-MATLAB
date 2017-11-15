% Creates 'count' number of neurons with the given width, using values
% from clusterData for their centers and random weights
function [neurons] = createNeurons(count, width, clusterData)
    [~, clusterCenters] = kmeans(clusterData, count, 'distance', 'sqEuclidean', 'start', 'sample');
    
%     [~, dayInYearClusters]   = kmeans(clusterData(:, 1), count);
%     [~, hourInDayClusters]   = kmeans(clusterData(:, 2), count);
%     [~, dayInWeekClusters]   = kmeans(clusterData(:, 3), count);

    % Create a neuron for each cluster, preallocating the array.
    neurons(count, 1) = Neuron();
    for i = 1 : size(clusterCenters, 1)
        neurons(i) = Neuron(clusterCenters(i), clusterCenters(i), ...
            clusterCenters(i), randomInRange(0, 1), width);
    end
    
%     % Create a neuron for each cluster, preallocating the array.
%     neurons(count, 1) = Neuron();
%     for i = 1 : size(dayInYearClusters)
%         neurons(i) = Neuron(dayInYearClusters(i), hourInDayClusters(i), ...
%             dayInWeekClusters(i), randomInRange(0, 1), width);
%     end
end