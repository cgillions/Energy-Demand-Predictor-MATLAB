% Returns a random number within the given range.
function random = randomInRange(min, max)
    random = min + (max - min) .* rand(1);
end