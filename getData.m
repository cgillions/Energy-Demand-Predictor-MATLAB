% Loads the data file and splits input from target.
function [x, target] = getData(filename)
    data   = load(filename);
    x      = data(:, 1:3);
    target = data(:, 4);
end