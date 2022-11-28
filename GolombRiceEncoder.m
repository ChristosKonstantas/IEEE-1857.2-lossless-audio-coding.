function code = GolombRiceEncoder(n,m)
%This is a golomb rice encoder so please mind to have a value of m that is
%a power of 2
q = floor(n/m);                   %compute the integer part of the quotient     
r=rem(n,m);                       %compute the reaminder of n/m

q1=ones(1,q);       
q_code=[q1 0];                    %unary code of quotient q
                                  %f,e used to check if m is a power of 2
if 2^(nextpow2(m))-m==0           %check whether m is a power of 2
    r_code=fliplr(de2bi(r,log2(m))); %log2(m)-bit binary code of r
else
    message='You can not use the Golomb-Rice encoder because m is not a power of 2';
    error(message)    
end
if(m~=1)
    code=[q_code,r_code];
else
    code=q_code;
end

end

