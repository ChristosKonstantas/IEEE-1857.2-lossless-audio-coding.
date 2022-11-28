function [encoded,k,bitstreamlength1] = expGolombEncoderFunc(seq, fileID)

% In this function, as proposed in forums and bibliography for the simple
% and non adaptive case of Exponential Golomb Rice coding we need to search
% exhaustively over the set to find the optimal k
% Reminder : the decoder only needs to know the encoded sequence and k.
%% INIT
ku=1;
j=0;
[block_length,total_blocks]=size(seq);
%% Find the maximum value k can take
while(1)
    j=j+1;
    ku=ku+1;
    if(2^(ku)>max(max(seq)))
        ku=ku-1;
        break
    end
end

minimum=99999;
k=1;
while k<= ku
    golomb_index=0;
    len=0;
    f=0;
    for i=1:total_blocks
        sequence=seq;
            for j2=1:block_length
                f=f+1;
                golomb_index=golomb_index+1;
                encoded_bitstream{golomb_index} = exp_Golomb(sequence(j2,i),k); 
                bitstreamlength(f) = length(encoded_bitstream{golomb_index});
%                 fwrite(fileID, encoded_bitstream{golomb_index},'ubit1');
                len=len+bitstreamlength(f);
            end
    end
    len=len/(total_blocks*block_length);
    if len < minimum
        minimum = len;
        minK=k;
    end
    k=k+1;
end
golomb_index=0;
%% At last, perform the encoding
for i=1:total_blocks
        sequence=seq;
            for j2=1:block_length
                golomb_index=golomb_index+1;
                encoded{golomb_index} = exp_Golomb(sequence(j2,i),minK);
                bitstreamlength1(golomb_index) = length(encoded{golomb_index});
                fwrite(fileID, encoded{golomb_index},'ubit1');
            end
end

k=minK;