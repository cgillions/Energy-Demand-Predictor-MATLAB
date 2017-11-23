% Loads the data file and splits input from target.
function [x, target] = getData(filename)
    data   = load(filename);
%     x      = data(:, 1:3);
%     target = data(:, 4);
    x      = data(1:2:end, 1:3);
    target = data(1:2:end, 4);
    
%     % Remove anomalies from the data set.
%     % https://uk.mathworks.com/matlabcentral/answers/298250-how-do-i-remove-outliers-from-a-matrix
%     disp("Initial size is: " + size(x, 1));
%     
%     % Find the row ids of outlying data points (central 2/3 section of the data (+-1/3)).
%     idx = bsxfun(@gt, target, mean(target) + 1.66 * std(target)) | bsxfun(@lt, target, mean(target) - 1.66 * std(target));
%     idx = any(idx, 2);
%     
%     % Remove those rows from the data set.
%     target(idx, :) = [];
%     x(idx, :)      = [];
%     
%     disp("Final size is: " + size(x, 1));
end