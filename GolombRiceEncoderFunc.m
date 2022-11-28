function [encoded,m,bitstreamlength1] = GolombRiceEncoderFunc(seq, fileID)

% In this function, as proposed in forums and bibliography for the simple
% and non adaptive case of Golomb Rice coding we need to search
% exhaustively over the set to find the optimal m 
% Reminder : the decoder only needs to know the encoded sequence and m.
%% INIT
mu=1;
j=0;
[block_length,total_blocks]=size(seq);
%% Find the maxmimum value m can take
while(1)
    j=j+1;
    mu=mu*2;
    if(mu>max(max(seq)))
        mu=mu/2;
        break
    end
end

%% Find m that gives the minimum length of encoded sequence
minimum=99999;
m=1;
while m<= mu
    golomb_index=0;
    len=0;
    k=0;
    for i=1:total_blocks
        sequence=seq;
            for j2=1:block_length
                k=k+1;
                golomb_index=golomb_index+1;
                encoded_bitstream{golomb_index} = GolombRiceEncoder(sequence(j2,i),m); %m=2^k
                bitstreamlength(k) = length(encoded_bitstream{golomb_index});
%                 fwrite(fileID, encoded_bitstream{golomb_index},'ubit1');
                len=len+bitstreamlength(k);
            end
    end
    len=len/(total_blocks*block_length);
    if len < minimum
        minimum = len;
        minM=m;
    end
    m=m*2;
end

golomb_index=0;
%% At last, perform the encoding
for i=1:total_blocks
        sequence=seq;
            for j2=1:block_length
                golomb_index=golomb_index+1;
                encoded{golomb_index} = GolombRiceEncoder(sequence(j2,i),minM); %m=2^k
                bitstreamlength1(golomb_index) = length(encoded{golomb_index});
                fwrite(fileID, encoded{golomb_index},'ubit1');
            end
end

m=minM;
end

