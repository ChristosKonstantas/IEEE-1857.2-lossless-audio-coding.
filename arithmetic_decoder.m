function decoded = arithmetic_decoder(code, counts, dec_length)
%%
%ARITHMETIC DECODER
%%
%INITIALIZATION FOR THE DECODER%

%cumulative frequencies matrix that represent the ranges of each symbol
CF = [0, cumsum(counts)];
%Total cumulative frequency T
T=CF(end);
%word length (2^m = 4*T)
m = ceil(log2(T)) + 2;
% Initialize the low and high range indices.
low = 0;
high = 2^m-1;
Scale3 = 0;
%Set the first m bits of code as a temporary tag
tag_b = code(1:m);
tag_d = bi2de(tag_b, 'left-msb');
%Initialize the decoded sequence
decoded = zeros(1,dec_length);
decodedIdx = 1;
i=m;
%%

%DECODING PROCESS%


%Basic decoder loop
while (decodedIdx <= dec_length)
    %compute range for the decodedIdx symbol
     w=(high+1)-low;
    % Compute scaled_symbol
     scaled_symbol =floor((T*(tag_d-low+1)-1)/w); 
    % Decode a symbol based on scaled_symbol
     % We wish to find where scaled_symbol is positioned
    if scaled_symbol == CF(end)
       symbol = length(CF)-1 ;
    end
    c = find(CF <= scaled_symbol);
    symbol = c(end);
    
    %Insert the decoded symbol to the decoded sequence
     decoded(decodedIdx) = symbol;
     decodedIdx = decodedIdx + 1;
    
    %compute CFlow and CFhigh
     CFlow=CF(symbol);
     CFhigh=CF(symbol+1);
    %compute the high and low
     high = low + floor(w*CFhigh/T) - 1;
     low = low + floor(w*CFlow/T);

   % E1, E2 and E3 conditions
     while(( isequal(bitget(low, m), bitget(high, m))) || (((isequal(bitget(low, m-1), 1)) && (isequal(bitget(high, m-1), 0) ) )))
        
        % Exit while loop if we have read all the input bitstream (code)
        if ( i==length(code))
            break
        end
  
        i=i+1;
        %-------------MIMIC THE ENCODING PROCEDURE-----------------------%
        
        %  if(E1 || E2) we know that the MSB is the same for both low and high range indices
        if isequal(bitget(low, m), bitget(high, m)) %if MSB of low = MSB of high

            % Multiplication by 2 is a left shift
            low = bitsll(low,1) + 0;
            
            high = bitsll(high,1) + 1; 
            
            % Left shift and read in code
            tag_d = bitsll(tag_d, 1) + code(i);
         % if(E3)   
        elseif ( (isequal(bitget(low, m-1), 1)) && (isequal(bitget(high, m-1), 0) ) )
            % Left shifts and update
            low = bitsll(low, 1) + 0;
            high  = bitsll(high,  1) + 1;
            tag_d = bitsll(tag_d, 1) + code(i);
            % Complement the new MSB of low, high and tag_d
            low = bitset(low, m, ~(bitget(low,m)));
            high = bitset(high, m, ~(bitget(high,m)));
            tag_d = bitset(tag_d, m, ~(bitget(tag_d,m)));            
        end
%%%%%%%%%%%%%%%% Restrict low, high and tag to be size m %%%%%%%%%%%%%%%%%%%%
        if(low >= 2^m)
         	low=de2bi(low);
         	low=low(1:m);
         	low=bi2de(low);
        end
            
        if( high >= 2^m)
            high=de2bi(high);
            high=high(1:m);
            high=bi2de(high);
        end
        
        if(tag_d >= 2^m)
         	tag_d=de2bi(tag_d);
         	tag_d=tag_d(1:m);
         	tag_d=bi2de(tag_d);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    end % end while
end % end while length(decoded)



end

