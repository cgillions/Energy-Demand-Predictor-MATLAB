% Plots the training and testing error for the network.
function plotError(epochRms, neuronWidth, neuronCount, learnBatch)
    figure;
    hold on;
    plot(1:size(epochRms, 1), epochRms(:, 1));
    plot(1:size(epochRms, 1), epochRms(:, 2));
    hold off;
    ylabel("RMS");
    xlabel("Epoch");
    legend(["Training error", "Testing error"]);
    
    if learnBatch
        batchLearn = "true";
    else
        batchLearn = "false";
    end
    title(sprintf("Network Error\nNeurons: %d, Width: %.2f, Batch learn: %s", neuronCount, neuronWidth, batchLearn));
end