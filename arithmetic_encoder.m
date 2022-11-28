function tag = arithmetic_encoder(seq, counts)
%%
%INITIALIZATION FOR THE ENCODER%

%counts for E3 condition
Scale3 = 0;
%cumulative frequencies matrix that represent the ranges of each symbol
CF = [0, cumsum(counts)];
%Total cumulative frequency T
T=CF(end);
%word length (2^m = 4*T)
m = ceil(log2(T)) +2;
%Initialize the low and high range indices.
low = 0;
high = 2^m-1;
% Tag index initialization
tag_index = 1;
%%

%ENCODING PROCESS%


% Loop for each symbol in the input sequence
for k = 1:length(seq)

    %compute range for the kth symbol
    w=(high+1)-low;
    %compute CFlow and CFhigh
    CFlow=CF(seq(k));
    CFhigh=CF(seq(k)+1);
    %compute the high and low
    high = low + floor(w*CFhigh/T) - 1;
    low = low + floor(w*CFlow/T);

     
     while(1)
        
        %  if(E1 || E2) we know that the MSB is the same for both low and high range indices
        if isequal(bitget(low, m), bitget(high, m)) %if MSB of low = MSB of high
            MSB = bitget(low, m); % get the MSB
            
            tag(tag_index) = MSB; % store the MSB
            
            tag_index = tag_index + 1; % update tag index
            
            % If E3 condition has happened transmit Scale3 times the complement of the current MSB
            while Scale3 > 0
                
                tag(tag_index) = ~MSB;
                
                tag_index = tag_index + 1;
                Scale3 = Scale3-1;
            end
%             if k == 1
%                 disp(low)  % <- set a breakpoint in this line DEBUG
%               end

            %Left shifts <<
          
            %By the time I've done my shifting I guarantee that the first
            %bit of low is 0 and the first bit of high is 1
            low = bitsll(low,1) + 0;
            high = bitsll(high,1) + 1; %we add 1 to ensure that high > low and keep the range as high as possible
            

%             if k == 1
%                 disp(low)  % <- set a breakpoint in this line DEBUG
%               end


        % if(E3)   
        elseif  (isequal(bitget(low, m-1), 1)) && (isequal(bitget(high, m-1), 0) )
            %Remember E3 state occurences.
            Scale3 = Scale3+1;
            %Complement the second MSB of low and high
            low = bitset(low, m-1, ~(bitget(low,m-1)));
            high = bitset(high, m-1, ~(bitget(high,m-1)));
            %Left shifts<<
            %By the time I've done my shifting I guarantee that the first
            %bit of low is 0 and the first bit of high is 1
            low = bitsll(low, 1) + 0;
            high  = bitsll(high, 1) + 1; %we add 1 to ensure that high > low and keep the range as high as possible

        else
            break;
        end
        
        %%%Restrict low and high to be size m%%%%
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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

%%
%If we want to terminate encoding it is important to be sure that the tag
%is greater than low
low = fliplr(de2bi(low, m));
if Scale3==0
    %Store low in m bits appended next to the tag
    tag(tag_index:tag_index+m-1) = low;
    tag_index = tag_index + m;
else
   %Store the MSB of low. 
   MSB = low(1);
   tag(tag_index) = MSB;
   tag_index = tag_index + 1;
   
   %Store the complemented MSBs because we still had E3 scales to do
   while(Scale3>0)
       tag(tag_index) = ~MSB;
       tag_index = tag_index + 1;
       Scale3=Scale3-1;
   end

   % Again store the m-1 (because we have already stored the MSB, otherwise
   % we store the m) bits of low and append them next to the tag so we are
   % sure that the tag is greater than low
   tag(tag_index:tag_index+m-2) = low(2:m);
   tag_index = tag_index + m - 1;
end          

%Store to the tag exactly the same values as the tag indices
tag = tag(1:tag_index-1);



end

