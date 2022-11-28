function G_Code = GolombEncoder(n,m)
%G(n,m)

q = floor(n/m);                   %quotient     
r=rem(n,m);                       %temainder
fl=floor(log2(m));
a=floor(log2(m)+1);
q1=ones(1,q);       
Unary_q=[q1 0];                    %UnaryCode(q)

if m==1              %if m=1, G(n,m)=UnaryCode(q)
    G_Code=Unary_q;
else
    if 2^(nextpow2(m))-m==0                %=0 only if m is a power of 2
        
        Binary_r=de2bi(r,log2(m),'left-msb'); %log2(m)-bit binary code of r
    else
        
        if r < (2^(a) - m) 
            
            Binary_r=de2bi(r,fl,'left-msb'); %b-bit binary representation of r
        else
            
            r=r+(2^(a) - m);
            Binary_r=de2bi(r,fl+1,'left-msb'); %b+1-bit binary representation of r
        end 
    end
    
    
G_Code=[Unary_q Binary_r];               %Output the Golomb Code 
end
