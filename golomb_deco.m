function decoded = GolombDecoder(G_Code, m)
% INIT
q = 0;
flag=1;
i=0;

%Stop when you see the first 0
while(flag)
    i=i+1;
    if G_Code(i) == 1
        q=q+1;
    else
        separator = i;   %Unary G_Code read
        flag=0;
    end
end

if (m == 1)   
    decoded = q;     
else    %Cases for the remainder
    
    BinaryCode = G_Code(separator+1:separator + floor(log2(m)));
    r = bi2de(BinaryCode,'left-msb');
    
    if r < (2^(floor(log2(m))+1) - m)
        separator = separator + floor(log2(m));
    else
        
        BinaryCode = G_Code(separator+1 : separator+floor(log2(m)+1));
        r = bi2de(BinaryCode,'left-msb') - (2^(floor(log2(m))+1) - m);
        separator = separator+floor(log2(m)+1);
    end
    
    decoded = q*m+r; %computing the symbol from the decoded quotient and remainder
end
