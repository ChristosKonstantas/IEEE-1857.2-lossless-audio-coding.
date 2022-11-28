function GR_Code = GolombEncoder(n,m)
%G(n,m)

q=floor(n/m);
q1=ones(1,q);   
Unary_q=[q1 0];     %UnaryCode(q)

r=rem(n,m);                
if m==1             %if m=1, G(n,m)=UnaryCode(q)
    GR_Code=Unary_q;
else
    if 2^(nextpow2(m))-m==0                %=0 only if m is a power of 2 
        Binary_r=fliplr(de2bi(r,log2(m))); %log2(m)-bit binary code of r
    else
        
        if r < (2^(floor(log2(m))+1) - m) 
            
            Binary_r= fliplr(de2bi(r,floor(log2(m)))); %log2(m)-bit binary representation of r
        else
            
            r= r+(2^(floor(log2(m))+1) - m);
            Binary_r= fliplr(de2bi(r,floor(log2(m))+1)); %log2(m)+1-bit binary representation of r
        end 
    end
    
    
GR_Code=[Unary_q Binary_r];               %Output the Golomb-Rice Code 
end
