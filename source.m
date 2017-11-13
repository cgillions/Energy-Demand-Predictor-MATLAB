function source
    % Seed the random number generator so experiments are comparable.
    rng(10);
    
    % Open the training and testing files.
    [trainX, trainTarget] = getData("train.csv");
    [testX,  testTarget]  = getData("test.csv");
    
    % Normalise the data.
    maxDayOfYear = 366;
    maxHour      = 24;
    maxDay       = 7;
    
    trainX(:, 1) = trainX(:, 1) / maxDayOfYear;
    trainX(:, 2) = trainX(:, 2) / maxHour;
    trainX(:, 3) = trainX(:, 3) / maxDay;
    
    testX(:, 1) = testX(:, 1) / maxDayOfYear;
    testX(:, 2) = testX(:, 2) / maxHour;
    testX(:, 3) = testX(:, 3) / maxDay;
    
    % Initialise network values.
    neuronCount  = 4;
    neuronWidth  = 0.2;
    learningRate = 0.2;
    learnBatch   = false;
    
    % Create the network.
    network = Network(learningRate, neuronCount, neuronWidth, trainX);
    
    % Train the network.
    % Keep track of the training & testing RMS error for each epoch.
    epochCount = 20;
    epochRms   = [epochCount, 2];
    for epoch  = 1:epochCount
        disp("Epoch " + epoch);
        % Get the network's output for the training & testing data set.
        [trainOutput, ~] = network.feedBatch(trainX);
        [testOutput,  ~] = network.feedBatch(testX);
        
        % Get the network's error from the training & testing output.
        testRms  = sqrt(sum(0.5 * (testTarget  - testOutput)  .^ 2)...
                    / size(testOutput,  1));
        trainRms = sqrt(sum(0.5 * (trainTarget - trainOutput) .^ 2)...
                    / size(trainOutput, 1));
        
        % Store the RMS.
        epochRms(epoch, :) = [trainRms, testRms]
        
        if learnBatch
            % Update the weights (batch).
            network.train(trainX, trainTarget);
        else
            % Update the weights for each data point (online).
            for dataIndex = 1:size(trainX, 1)
                network.train(trainX(dataIndex, :), trainTarget(dataIndex, :));
            end
        end
    end
    
    % Test the trained network.
    [trainOutput, ~] = network.feedBatch(trainX);
    [testOutput,  ~] = network.feedBatch(testX);
    
    % Generate many values of x across the domain of the input in order 
    % to plot the network's function.
    figure;
    subplot(1, 3, 1);
    plot(trainX(:, 1) * maxDayOfYear, trainOutput, "ro");
    hold on;
    plot(trainX(:, 1) * maxDayOfYear, trainTarget, "b*");
    hold off;
    
    subplot(1, 3, 2);
    plot(trainX(:, 2) * maxHour, trainOutput, "ro");
    hold on;
    plot(trainX(:, 2) * maxHour, trainTarget, "b*");
    hold off;
    
    subplot(1, 3, 3);
    plot(trainX(:, 3) * maxDay, trainOutput, "ro");
    hold on;
    plot(trainX(:, 3) * maxDay, trainTarget, "b*");
    hold off;
    
    % Plot the network's error per epoch.
    plotError(epochRms, neuronWidth, neuronCount, learnBatch);
end

