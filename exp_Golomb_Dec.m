function decoded = exp_Golomb_Dec(inpt,k,len,total)
idx=0;
arrayIndex=1;
while 1
    % Count the number of 1's followed by the first 0 
    idx=idx+1;
    q = 0;
        for i = arrayIndex:length(inpt)
            if inpt(i) == 1
                q = q + 1; %count the number of 1s-> q is counted here
            else
                ptr = i;   % first 0
                break;
            end
        end
        if(idx>len*total)
            break;
        end
        arrayIndex=ptr;
        % decoding the remainder
         r_code = inpt(arrayIndex+1:arrayIndex+q+k);
         r = bi2de(r_code,'left-msb');
         arrayIndex=arrayIndex+q+k+1;
  
         if (k == 0)
         	decoded(idx) = q;   % special case for m = 1
         else
          %  su=0;
           % for j=0:q-1
            %    su=su+2^(j+k);
            %end 
         decoded(idx) = r+(2^k)*(2^q-1); %computing the symbol from the decoded quotient and remainder
         end

end

