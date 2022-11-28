function [blocks, total_blocks] = Framing(input, block_length)
total_samples = length(input);
total_blocks= floor(total_samples/block_length);
blocks = zeros(total_blocks, block_length);

%Blocking the input audio data
for i = 1:total_blocks
    for j = 1:block_length
        blocks(i,j) = input((i-1)*block_length + j);
    end
end
blocks=blocks' ;

end

