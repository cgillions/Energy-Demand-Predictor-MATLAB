function source
    % Seed the random number generator so experiments are comparable.
    rng(10);
    
    % Open the training and testing files.
    [trainX, trainTarget] = getData("train.csv");
    [testX,  testTarget]  = getData("test.csv");
    
    % Normalise the data between 0 and 1.
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
    neuronCount  = 200;
    neuronWidth  = 0.05;
    learningRate = 0.01;
    learnBatch   = false;
    
    % Create the network.
    network = Network(learningRate, neuronCount, neuronWidth, trainX);
    
    % Train the network.
    % Keep track of the training & testing RMS error for each epoch.
    epochCount = 200;
    epochRms   = zeros(epochCount, 2);
    for epoch  = 1:epochCount
        disp("Epoch " + epoch);
        % Get the network's output for the training & testing data set.
        [trainOutput, ~] = network.feedBatch(trainX);
        [testOutput,  ~] = network.feedBatch(testX);
        
        % Get the network's error from the training & testing output.
        testRms  = sqrt(sum((testTarget  - testOutput)  .^ 2)...
                    / size(testOutput,  1));
        trainRms = sqrt(sum((trainTarget - trainOutput) .^ 2)...
                    / size(trainOutput, 1));
        
        % Store the RMS.
        epochRms(epoch, :) = [trainRms, testRms];
        
        % Check if we've converged on a solution.
        if (epoch > 2) && (epochRms(epoch-1, 1) - trainRms < 0.0001) && ...
                (epochRms(epoch-2, 1) - trainRms < 0.0001)
            break;
        end        
        
        if learnBatch
            % Update the weights (batch).
            network.train(trainX, trainTarget);
        else
            % Update the weights for each data point (online).
            for dataIndex = 1:size(trainX, 1)
                network.train(trainX(dataIndex, :), trainTarget(dataIndex));
            end
        end
    end
    
    % Plot the network's error per epoch.
    plotError(epochRms, neuronWidth, neuronCount, learnBatch);
    
    % Validate the trained network.
    [validationInput, validationTarget] = getData("validation.csv");
    
    % Normalise the validation data.
    validationInput(:, 1) = validationInput(:, 1) / maxDayOfYear;
    validationInput(:, 2) = validationInput(:, 2) / maxHour;
    validationInput(:, 3) = validationInput(:, 3) / maxDay;
    
    % Get the output.
    validationOutput = network.feedBatch(validationInput);
    
    % Plot the output from the validation data.
    figure;
    subplot(3, 1, 1);
    hold on;
    plot(validationInput(:, 1) * maxDayOfYear, validationOutput, "bo");
    plot(validationInput(:, 1) * maxDayOfYear, validationTarget, "rx");
    hold off;
    title("Demand per day of year");
    xlim([0, 366]);
    
    subplot(3, 1, 2);
    hold on;
    plot(validationInput(:, 2) * maxHour, validationOutput, "bo");
    plot(validationInput(:, 2) * maxHour, validationTarget, "rx");
    hold off;
    title("Demand per hour of day");
    xlim([0, 24]);
    
    subplot(3, 1, 3);
    hold on;
    plot(validationInput(:, 3) * maxDay, validationOutput, "bo");
    plot(validationInput(:, 3) * maxDay, validationTarget, "rx");
    hold off;
    title("Demand per day of week");
    xlim([1, 7]);
    
    figure;
    colors = [255,   0,   0; 
              255, 125,   0; 
              255, 255,   0; 
                0, 255,   0;
                0, 255, 255;
                0,   0, 255;
                255, 0, 255];
            
    inputDayColors = colors(validationInput(:, 3) * maxDay, :);
    scatter3(validationInput(:, 1) * maxDayOfYear, validationInput(:, 2) * maxHour, validationOutput, validationInput(:, 3) * maxDay, inputDayColors);
end

