function decoded = GolombRiceDecoder(block_length,total_blocks,inpt1,m)
%GolombRiceDecoder decompresses all the input file
idx=1;
arrayIndex=1;

while 1  
    q = 0;
        for i = arrayIndex:length(inpt1)

            if inpt1(i) == 1
                q = q + 1; %count the number of 1s-> q is counted here
                separator=q;
            else
                separator = i;   % first 0 is the separator after prefix
                break;
            end
        end

            if(idx==block_length*total_blocks+1)
                break;
            end
        
        
        if (m == 1)
         	decoded(idx) = q;   % special case for m = 1
        else
            arrayIndex=separator; %Array index shows the position we are currently on the array we are reading
            % decoding the remainder
            r_code = inpt1(arrayIndex+1:arrayIndex+log2(m));
            arrayIndex=arrayIndex+(log2(m)+1);
            r = bi2de(r_code,'left-msb');%suffix
            decoded(idx) = q * m + r; %computing the symbol from the decoded quotient and remainder
        end
    idx=idx+1;
end




end

